# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace Language Model API."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Optional
import huggingface_hub
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib, sampling
from concordia.utils import text


DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
MAX_TOKENS_FOR_CHOICE = 32

_DEFAULT_SYSTEM_MESSAGE = (
    'Continue the user\'s sentences. Never repeat their starts. Do not repeat yourself either. For example, '
    'when you see \'Bob is\', you should continue the sentence after '
    'the word \'is\'. Here are some more examples: \'Question: Is Jake a '
    'turtle?\nAnswer: Jake is \' should be completed as \'not a turtle.\' and '
    '\'Question: What is Priya doing right now?\nAnswer: Priya is currently \' '
    'should be completed as \'working on repairing the sink.\'. Notice that '
    'it is OK to be creative with how you finish the user\'s sentences. The '
    'most important thing is to always continue in the same style as the user.'
)


class HuggingFaceLanguageModel(language_model.LanguageModel):
  """Language model API for HuggingFace models.

  This class supports models loaded via the `transformers` library,
  which can be either from the HuggingFace Hub or a local path.
  """

  def __init__(
      self,
      model_name: str,
      *,
      device: int,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = 'huggingface_language_model',
      max_tokens_for_choice: int = MAX_TOKENS_FOR_CHOICE,
      api_key: str | None = None,
      use_bnb_4bit: bool = False,
  ):
    """Initializes the HuggingFace language model.

    Args:
      model_name: A descriptive name for the model. If model_path is None,
        this is also used as the HuggingFace model identifier.
      model_path: The HuggingFace model identifier (e.g., 'gpt2') or path to a
        local model. If None, model_name is used.
      device: The device to run the model on (e.g., 'cpu', 'cuda', 'auto').
        Passed to the `device_map` argument of `transformers.pipeline`.
      verbose: Whether to print verbose output.
      measurements: The measurements object to log telemetry.
      channel: The channel to use for logging measurements.
      max_tokens_for_choice: Max new tokens to generate when sampling a choice.
      api_key: The API key for the HuggingFace model.
    """
    huggingface_hub.login(token=api_key)
    self._model_identifier = model_name
    self._device = f'cuda:{device}'
    self._measurements = measurements
    self._channel = channel
    self._max_tokens_for_choice = max_tokens_for_choice

    # Load tokenizer
    self._tokenizer = AutoTokenizer.from_pretrained(self._model_identifier, use_fast=True, trust_remote_code=True)
    if not self._tokenizer.pad_token:
      self._tokenizer.pad_token = self._tokenizer.eos_token
      self._tokenizer.pad_token_id = self._tokenizer.eos_token_id


    if use_bnb_4bit:
      bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
      )
      model = AutoModelForCausalLM.from_pretrained(
        self._model_identifier,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    else:
      bnb_config = None
      model = model_name

    # Load pipeline
    self._pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=self._tokenizer,
    )


  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = DEFAULT_TIMEOUT_SECONDS, # pylint: disable=unused-argument
      seed: int | None = None, # pylint: disable=unused-argument
  ) -> tuple[str, Optional[Mapping[str, Any]]]:
    if temperature == 0:
      generation_kwargs = {"do_sample": False}
    elif temperature > 0:
      generation_kwargs = {
          "temperature": temperature,
          "do_sample": True,
          "top_p": 0.9
      }
    else:
      generation_kwargs = {"do_sample": False}

    full_prompt = f"{_DEFAULT_SYSTEM_MESSAGE}\n\n{prompt}"

    try:
      response = self._pipeline(
          full_prompt,
          max_new_tokens=max_tokens,
          pad_token_id=self._tokenizer.pad_token_id,
          **generation_kwargs
      )

      if response and isinstance(response, list) and \
         isinstance(response[0], dict) and 'generated_text' in response[0]:
        full_text = response[0]['generated_text']

        if full_text.startswith(full_prompt):
          generated_text_only = full_text[len(full_prompt):]
        else:
          generated_text_only = full_text

        truncated_text = text.truncate(generated_text_only, delimiters=terminators)

        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel,
              {'raw_text_length': len(truncated_text),
               'model_identifier': self._model_identifier})
        return truncated_text, None

      else:
        return "", None

    except Exception as e: # pylint: disable=broad-except
      raise e



  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *_,
      seed: int | None = None, # pylint: disable=unused-argument
  ) -> tuple[int, str, Mapping[str, Any]]:
    for attempts in range(MAX_MULTIPLE_CHOICE_ATTEMPTS):
      current_temperature = sampling.dynamically_adjust_temperature(
          attempts, MAX_MULTIPLE_CHOICE_ATTEMPTS
      )
      sample, _ = self.sample_text(
          prompt=prompt,
          max_tokens=self._max_tokens_for_choice,
          temperature=current_temperature,
          terminators=['\)', '\n', ' ', '.', ',']
      )

      answer = sampling.extract_choice_response(sample)

      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts}
          )
        debug = {}
        return idx, responses[idx], debug

    raise language_model.InvalidResponseError(f"Too many multiple choice attempts.\nLast attempt: {sample}, extracted: {answer}")
