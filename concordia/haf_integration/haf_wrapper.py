from concordia.language_model.huggingface_model import HuggingFaceLanguageModel
from concordia.typing import entity as entity_lib
from concordia.haf_integration.justify_stage.justify import HAFJustifyStage
from concordia.haf_integration.uphold_reasons_stage.uphold_reasons import HAFUpholdReasonStage
from concordia.haf_integration.uphold_stance_stage.uphold_stance import HAFUpholdStanceStage


class HAFWrapper:
  """Wrapper that runs all three HAF stages consecutively.

  This orchestrates the complete HAF evaluation pipeline:
  1. Justify Stage: Generates decision with reasons
  2. Uphold Reasons Stage: Evaluates if reasons are complete
  3. Uphold Stance Stage: Evaluates if each reason is individually sufficient

  Returns the justify stage output so simulation proceeds normally.
  """

  def __init__(
      self,
      agent_name: str,
      current_time_str: str,
      action_spec: entity_lib.ActionSpec,
      general_guidance_text: str,
      model: HuggingFaceLanguageModel,
  ):
    self.agent_name = agent_name
    self.current_time_str = current_time_str
    self.action_spec = action_spec
    self.general_guidance_text = general_guidance_text
    self.model = model

  def run(self) -> dict:
    """Execute all three HAF stages consecutively.

    Returns:
      Dictionary with justify stage results (to maintain simulation flow).
      Other stage results are logged/saved but not returned.
    """
    justify_stage = HAFJustifyStage(
        agent_name=self.agent_name,
        current_time_str=self.current_time_str,
        action_spec=self.action_spec,
        general_guidance_text=self.general_guidance_text,
        model=self.model,
    )
    justify_result = justify_stage.run()

    decision = justify_result['Returned Action String'].split('DECISION:')[1].split('REASON(S):')[0].strip()

    with open(justify_stage.output_file_path.replace('justify_stage_data.pkl', f'justify_stage_data_{self.current_time_str}_{self.agent_name}.pkl'), 'rb') as f:
      import pickle
      justify_data = pickle.load(f)
      reasons = justify_data['reasons']

    uphold_reasons_stage = HAFUpholdReasonStage(
        agent_name=self.agent_name,
        current_time_str=self.current_time_str,
        action_spec=self.action_spec,
        general_guidance_text=self.general_guidance_text,
        model=self.model,
        decision=decision,
        reasons=reasons,
    )
    uphold_reasons_stage.run()

    uphold_stance_stage = HAFUpholdStanceStage(
        agent_name=self.agent_name,
        current_time_str=self.current_time_str,
        action_spec=self.action_spec,
        general_guidance_text=self.general_guidance_text,
        model=self.model,
        decision=decision,
        reasons=reasons,
    )
    uphold_stance_stage.run()

    return justify_result
