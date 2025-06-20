# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the  License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language Model that uses HuggingFace's transformers library."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import huggingface_hub

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from typing_extensions import override


DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Complete the user's sentence concisely and"
    ' appropriately. Avoid unnecessary elaboration or flowery language. Keep'
    ' responses brief and to the point.'
)


class HuggingFaceLanguageModel(language_model.LanguageModel):
  """Language Model that uses HuggingFace's transformer models."""

  def __init__(
      self,
      model_name: str,
      api_key: str,
      *,
      device: int = 0,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      dtype: str = 'bfloat16',
  ):
    """Initializes the instance.

    Args:
      model_name: The name of the HuggingFace model to use.
        (e.g., "gpt2", "distilgpt2").
      api_key: The API key to use when accessing the HuggingFace API. If None, will
        use the HF_API_KEY environment variable.
      device: The device to run the model on. -1 for CPU, 0 for GPU 0, etc.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._api_key = api_key
    self._device = device

    huggingface_hub.login(api_key)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=dtype,
    )

    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    if self.tokenizer.pad_token_id is None:
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
    )

    self._pipeline = pipeline(
        'text-generation',
        model=self.model,
        tokenizer=self.tokenizer,
        batch_size=1,
        pad_token_id=self.tokenizer.pad_token_id,
        framework='pt',
        trust_remote_code=True,
    )
    print(
        'DEBUG: HuggingFace Pipeline initialized on device:'
        f' {self._pipeline.device}'
    )

  @property
  def device(self):
    return self.model.device

  def generate(self, **kwargs):
    return self.model.generate(**kwargs)

  @override
  def sample_text(
      self,
      prompt: str,
      *_,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      seed: int | None = None,
  ) -> str:
    """Samples text from the HuggingFace model."""
    if temperature <= 0:
      temperature = 0.1

    messages = [
        {'role': 'system', 'content': DEFAULT_SYSTEM_MESSAGE},
        {
            'role': 'user',
            'content': 'Question: Is Jake a turtle?\\nAnswer: Jake is ',
        },
        {'role': 'assistant', 'content': 'not a turtle.'},
        {'role': 'user', 'content': prompt},
    ]

    try:
      generation_args = {
          'max_new_tokens': min(max_tokens, 50),
          'temperature': min(temperature, 0.3),
          'return_full_text': False,
          'num_return_sequences': 1,
          'do_sample': True,
          'repetition_penalty': 1.3,
          'no_repeat_ngram_size': 4,
          'length_penalty': 0.8,
          'pad_token_id': self.tokenizer.pad_token_id,
      }
      if seed is not None:
        generation_args['seed'] = seed

      outputs = self._pipeline(messages, **generation_args)
      generated_text = outputs[0]['generated_text']

      if terminators:
        for term in terminators:
          if term in generated_text:
            generated_text = generated_text.split(term)[0]

    except Exception as e:
      print(f'Error during HuggingFace model generation: {e}')
      return ''

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_text)},
      )
    return generated_text

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *_,
      seed: int | None = None,  # pylint: disable=unused-argument
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Samples a response from those available using the HuggingFace model.

    This is a naive implementation. A more sophisticated approach would involve
    scoring each response option based on the model's likelihood (e.g., using
    conditional probabilities if the model/pipeline supports it easily, or by
    looking at perplexity of `prompt + response`).

    For now, we'll just generate text and see if it happens to match one of the
    responses, or try to steer the model. This is not ideal for multiple choice.

    Args:
      prompt: the initial text to condition on.
      responses: the responses to choose from.
      seed: optional seed for the sampling.

    Returns:
      (index, response, info). The index of the sampled response, the sampled
      response, and some info about the sampling process.
    """
    # Attempt to format prompt to guide model towards a choice.
    choice_prompt = f'{prompt}\\nWhich of the following is correct?\\n'
    for i, r in enumerate(responses):
      choice_prompt += f'{i+1}. {r}\\n'
    choice_prompt += (
        'Answer with the number and the text of the correct option:'
    )

    MAX_ATTEMPTS = 3
    attempts = 0
    temperature = language_model.DEFAULT_TEMPERATURE

    while attempts < MAX_ATTEMPTS:
      generated_text = self.sample_text(
          choice_prompt,
          temperature=temperature,
          max_tokens=max(len(r) for r in responses)
          + 20,  # Give some room for choice text
          seed=seed + attempts
          if seed is not None
          else None,  # Vary seed per attempt
      )

      for idx, response_text in enumerate(responses):
        if response_text in generated_text:
          if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel, {'choices_calls': attempts + 1}
            )
          return (
              idx,
              response_text,
              {'attempt': attempts + 1, 'raw_sample': generated_text},
          )

      attempts += 1
      temperature += 0.2  # Increase temperature for next attempt

    # If no choice is matched, raise an error or return a default.
    raise language_model.InvalidResponseError(
        'HuggingFace model failed to select a valid choice after'
        f" {MAX_ATTEMPTS} attempts. Last sample: '{generated_text}' for prompt:"
        f" '{choice_prompt}'"
    )
