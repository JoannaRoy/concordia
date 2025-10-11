import pickle
import torch
import os
from concordia.document import interactive_document
from concordia.language_model.huggingface_model import HuggingFaceLanguageModel
from concordia.typing import entity as entity_lib
from concordia.haf_integration.haf_utils import BASE_PATH, get_inputs_and_generations, get_final_output_string, get_logits_data, parse_numbered_reasons

UPHOLD_REASON_INSTRUCTIONS = """\
Given the above decision and the reasons provided, determine if additional reasons are required to fully justify the decision.
Follow this format for your response:
COMPLETE: [Yes/No - whether the current reasons are sufficient or if more are needed]
ADDITIONAL_REASON(S): [If not complete, provide additional reasons in a numbered list. Otherwise, state "None."]"""


class HAFUpholdReasonStage:
  """Handles the uphold-reason stage of HAF evaluation.

  This stage evaluates if the entire set of reasons from the justify stage
  is complete or if additional reasons are needed to fully justify the decision.
  """

  def __init__(
      self,
      agent_name: str,
      current_time_str: str,
      action_spec: entity_lib.ActionSpec,
      general_guidance_text: str,
      model: HuggingFaceLanguageModel,
      decision: str,
      reasons: dict[str, torch.Tensor],
      output_file_path: str = f'{BASE_PATH}/uphold_reasons_stage/data/uphold_reasons_stage_data.pkl'
  ):
    self.agent_name = agent_name
    self.current_time_str = current_time_str
    self.action_spec = action_spec
    self.general_guidance_text = general_guidance_text
    self.model = model
    self.decision = decision
    self.reasons = reasons
    self.output_file_path = output_file_path

  def format_prompt(self, answer_prefix: str) -> str:
    prompt = interactive_document.InteractiveDocument(self.model)
    prompt.statement(
        f'General reasoning guidance for {self.agent_name}:\n{self.general_guidance_text}'
    )

    call_to_action = self.action_spec.call_to_action.format(
        name=self.agent_name
    )
    prompt.statement(f'Original Question: {call_to_action}\n')
    prompt.statement(f'Decision Made: {self.decision}\n')

    reasons_list = '\n'.join([f'{i+1}. {reason}' for i, reason in enumerate(self.reasons.keys())])
    prompt.statement(f'Reasons Provided:\n{reasons_list}\n')

    final_prompt = f'{UPHOLD_REASON_INSTRUCTIONS}'
    prompt.statement(f'Question: {final_prompt}\n')
    prompt.statement(f'Answer: {answer_prefix}')
    return prompt.view().text()

  def _parse_completeness(self, generation_string: str) -> bool:
    """Parse whether the reason set is complete from the output."""
    if 'COMPLETE:' not in generation_string:
      return False

    completeness_text = generation_string.split('COMPLETE:')[1].split('ADDITIONAL_REASON(S):')[0].strip()
    return 'yes' in completeness_text.lower()

  def save_uphold_reasons_stage(self, is_complete: bool, additional_reasons: dict, logits_data: dict, prompt: str, output: str) -> None:
    data = {
        'agent_name': self.agent_name,
        'input_text': self.general_guidance_text,
        'decision': self.decision,
        'original_reasons': list(self.reasons.keys()),
        'is_complete': is_complete,
        'additional_reasons': additional_reasons,
        'output_tokens': logits_data['output_tokens'],
        'logits': logits_data['logits'],
        'scores': logits_data['scores'],
        'prompt': prompt,
        'output': output,
    }

    file_path = self.output_file_path.replace('uphold_reasons_stage_data.pkl', f'uphold_reasons_stage_data_{self.current_time_str}_{self.agent_name}.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
      pickle.dump(data, f)

  def run(self) -> dict:
    prompt = self.format_prompt(f'{self.agent_name} ')
    inputs, generations = get_inputs_and_generations(self.model, prompt)

    output_string = get_final_output_string(
        self.model, inputs, generations, f'{self.agent_name} '
    )

    logits_data = get_logits_data(self.model, inputs, generations)
    is_complete = self._parse_completeness(logits_data['generation_string'])
    additional_reasons = parse_numbered_reasons(
        logits_data['generation_string'],
        'ADDITIONAL_REASON(S):',
        self.model.tokenizer
    )

    self.save_uphold_reasons_stage(is_complete, additional_reasons, logits_data, prompt, output_string)

    return {
        'Key': f'{self.agent_name} - Uphold Reasons Step',
        'Action Type': self.action_spec.output_type.name,
        'Skipped': False,
        'Is Complete': is_complete,
        'Additional Reasons': list(additional_reasons.keys()),
    }
