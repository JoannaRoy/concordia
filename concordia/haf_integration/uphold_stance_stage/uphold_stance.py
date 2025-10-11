import pickle
import torch
import os
from concordia.document import interactive_document
from concordia.language_model.huggingface_model import HuggingFaceLanguageModel
from concordia.typing import entity as entity_lib
from concordia.haf_integration.haf_utils import BASE_PATH, get_inputs_and_generations, get_final_output_string, get_logits_data

UPHOLD_STANCE_INSTRUCTIONS = """\
Given the above context and the specific reason provided, evaluate if this single reason alone is sufficient to justify the decision.
Follow this format for your response:
SUFFICIENCY: [Yes/No - whether this reason alone is sufficient to justify the decision]"""


class HAFUpholdStanceStage:
  """Handles the uphold-stance stage of HAF evaluation.

  This stage evaluates if each individual reason from the justify stage is
  sufficient on its own to support the decision (individual sufficiency).
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
      output_file_path: str = f'{BASE_PATH}/uphold_stance_stage/data/uphold_stance_stage_data.pkl'
  ):
    self.agent_name = agent_name
    self.current_time_str = current_time_str
    self.action_spec = action_spec
    self.general_guidance_text = general_guidance_text
    self.model = model
    self.decision = decision
    self.reasons = reasons
    self.output_file_path = output_file_path

  def format_prompt(self, reason_text: str, answer_prefix: str) -> str:
    prompt = interactive_document.InteractiveDocument(self.model)
    prompt.statement(
        f'General reasoning guidance for {self.agent_name}:\n{self.general_guidance_text}'
    )

    call_to_action = self.action_spec.call_to_action.format(
        name=self.agent_name
    )
    prompt.statement(f'Original Question: {call_to_action}\n')
    prompt.statement(f'Decision Made: {self.decision}\n')
    prompt.statement(f'Reason Being Evaluated: {reason_text}\n')

    final_prompt = f'{UPHOLD_STANCE_INSTRUCTIONS}'
    prompt.statement(f'Question: {final_prompt}\n')
    prompt.statement(f'Answer: {answer_prefix}')
    return prompt.view().text()

  def _parse_sufficiency(self, generation_string: str) -> bool:
    """Parse whether the reason is sufficient from the output."""
    if 'SUFFICIENCY:' not in generation_string:
      return False

    sufficiency_text = generation_string.split('SUFFICIENCY:')[1].split('ADDITIONAL_REASON(S):')[0].strip()
    return 'yes' in sufficiency_text.lower()

  def save_uphold_stance_stage(self, reason_evaluations: list[dict]) -> None:
    data = {
        'agent_name': self.agent_name,
        'input_text': self.general_guidance_text,
        'decision': self.decision,
        'original_reasons': list(self.reasons.keys()),
        'reason_evaluations': reason_evaluations,
    }

    file_path = self.output_file_path.replace('uphold_stance_stage_data.pkl', f'uphold_stance_stage_data_{self.current_time_str}_{self.agent_name}.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
      pickle.dump(data, f)

  def run(self) -> dict:
    reason_evaluations = []

    for reason_text in self.reasons.keys():
      prompt = self.format_prompt(reason_text, f'{self.agent_name} ')
      inputs, generations = get_inputs_and_generations(self.model, prompt)

      output_string = get_final_output_string(
          self.model, inputs, generations, f'{self.agent_name} '
      )

      logits_data = get_logits_data(self.model, inputs, generations)
      is_sufficient = self._parse_sufficiency(logits_data['generation_string'])

      reason_evaluations.append({
          'original_reason': reason_text,
          'is_sufficient': is_sufficient,
          'output_tokens': logits_data['output_tokens'],
          'logits': logits_data['logits'],
          'scores': logits_data['scores'],
          'prompt': prompt,
          'output': output_string,
      })

    self.save_uphold_stance_stage(reason_evaluations)

    return {
        'Key': f'{self.agent_name} - Uphold Stance Step',
        'Action Type': self.action_spec.output_type.name,
        'Skipped': False,
        'Num Reasons Evaluated': len(reason_evaluations),
    }
