import pickle
import numpy as np
import torch
import os
from concordia.document import interactive_document
from concordia.language_model.huggingface_model import HuggingFaceLanguageModel
from concordia.typing import entity as entity_lib
from concordia.haf_integration.haf_utils import BASE_PATH, get_inputs_and_generations, get_reason_entropies, get_final_output_string, get_logits_data, parse_numbered_reasons



REASONING_INSTRUCTIONS = """\
When making your decision, you must first clearly state your decision, then provide specific reason(s) for it.
If there is more than one reason, provide them in a numbered list.
Follow this format for your response:
DECISION: [Your decision]
REASON(S): [Your reasons (e.g., 1. Reason one. 2. Reason two.)]"""


class HAFJustifyStage:
  """Handles the justification stage of HAF (Human-Aligned Faithfulness) decision making.

  This class generates structured decisions with reasoning, processes model outputs,
  and extracts token-wise entropies for analysis.
  """

  def __init__(
      self,
      agent_name: str,
      current_time_str: str,
      action_spec: entity_lib.ActionSpec,
      general_guidance_text: str,
      model: HuggingFaceLanguageModel,
      output_file_path: str = f'{BASE_PATH}/justify_stage/data/justify_stage_data.pkl'
  ):
    """Initialize the HAF justify stage.

    Args:
      agent_name: Name of the agent making the decision.
      current_time_str: Current timestamp as string.
      action_spec: Specification for the action to be taken.
      general_guidance_text: General guidance context for decision making.
      model: HuggingFace language model to use for generation (whichever is being used for the simulation).
      output_file_path: Base path for output JSON files.
    """
    self.agent_name = agent_name
    self.current_time_str = current_time_str
    self.action_spec = action_spec
    self.general_guidance_text = general_guidance_text
    self.model = model
    self.output_file_path = output_file_path

  def format_prompt(self, decision_answer_prefix: str) -> str:
    """Format the prompt for decision generation.

    Args:
      decision_answer_prefix: Prefix to use for the model's answer.

    Returns:
      Formatted prompt string.
    """
    decision_prompt = interactive_document.InteractiveDocument(self.model)
    decision_prompt.statement(
        f'General reasoning guidance for {self.agent_name}:\n{self.general_guidance_text}'
    )

    call_to_action_for_decision = self.action_spec.call_to_action.format(
        name=self.agent_name
    )
    final_call_to_action_prompt_text = (
        f'{call_to_action_for_decision}\n\n{REASONING_INSTRUCTIONS}'
    )

    decision_prompt.statement(f'Question: {final_call_to_action_prompt_text}\n')
    decision_prompt.statement(f'Answer: {decision_answer_prefix}')
    return decision_prompt.view().text()


  def save_justify_stage(self, reasons, logits_data, reason_logits_entropies, reason_scores_entropies, decision_output_string: str) -> None:
    """Save processed data to JSON file.

    Args:
      reasons: Dictionary of reasons and their token encodings {reason: token_encoding}.
      reason_logits_entropies: List of logit entropies for each reason.
      reason_scores_entropies: List of score entropies for each reason.
      decision_output_string: Full decision output string.
    """
    decision = decision_output_string.split('DECISION:')[1].split('REASON(S):')[0]

    data = {
        'agent_name': self.agent_name,
        'input_text': self.general_guidance_text,
        'output_tokens': logits_data['output_tokens'],
        'decision': decision.strip(),
        'reasons': reasons,
        'reason_logits_entropies': reason_logits_entropies,
        'reason_scores_entropies': reason_scores_entropies,
    }

    file_path = self.output_file_path.replace('justify_stage_data.pkl', f'justify_stage_data_{self.current_time_str}_{self.agent_name}.pkl')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
      pickle.dump(data, f)


  def run(self) -> dict:
    """Execute the complete HAF justify stage process.

    Returns:
      Dictionary containing log entry with all processing results.
    """
    prompt = self.format_prompt(f'{self.agent_name} ')
    inputs, generations = get_inputs_and_generations(self.model, prompt)

    decision_output_string = get_final_output_string(
        self.model, inputs, generations, f'{self.agent_name} '
    )

    logits_data = get_logits_data(self.model, inputs, generations)
    reasons = parse_numbered_reasons(logits_data['generation_string'], 'REASON(S):', self.model.tokenizer)
    reason_logits_entropies, reason_scores_entropies = get_reason_entropies(
        logits_data['logits'],
        logits_data['scores'],
        reasons,
        inputs['input_ids'],
        logits_data['output_tokens']
    )

    self.save_justify_stage(reasons, logits_data, reason_logits_entropies, reason_scores_entropies, decision_output_string)

    return {
        'Key': f'{self.agent_name} - Decision Step',
        'Action Type': self.action_spec.output_type.name,
        'Skipped': False,
        'Input General Guidance': self.general_guidance_text,
        'LLM Call for Decision (question)': prompt,
        'LLM Call for Decision (answer_prefix)': f'{self.agent_name} ',
        'LLM Call for Decision (raw_output)': generations,
        'Returned Action String': decision_output_string,
        'Chain of thought (Decision Step)': prompt.splitlines(),
    }
