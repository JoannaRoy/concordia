import pickle
import re
import torch
import numpy as np

BASE_PATH = 'concordia/haf_integration/'

SAFTE_CONFIG = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 1.0,
    'max_new_tokens': 100,
    'return_dict_in_generate': True,
    'output_scores': True,
    'output_logits': True,
}

def pkl_to_dict(pkl_file_path: str) -> dict:
  with open(pkl_file_path, 'rb') as f:
    return pickle.load(f)

def extract_decision_time_from_file_name(justify_stage_file_path: str) -> str:
  return justify_stage_file_path.split('_')[-2]

def preview_metrics(metrics_file_path: str) -> None:
  metrics = pkl_to_dict(metrics_file_path)
  print(metrics)

def get_inputs_and_generations(model, prompt: str, max_new_tokens: int = 200):
  config = SAFTE_CONFIG.copy()
  config['pad_token_id'] = model.tokenizer.pad_token_id
  config['eos_token_id'] = model.tokenizer.eos_token_id
  config['max_new_tokens'] = max_new_tokens

  inputs = model.tokenizer(
      prompt, return_tensors='pt', padding=True
  ).to(model.device)

  generations = model.generate(**inputs, **config)
  return inputs, generations

def get_logits_data(model, inputs, generations) -> dict:
  return {
      'input_tokens': inputs['input_ids'].to('cpu'),
      'output_tokens': generations['sequences'].to('cpu')[0],
      'logits': torch.stack(generations['logits'], dim=1).to('cpu'),
      'scores': torch.stack(generations['scores'], dim=1).to('cpu'),
      'generation_string': model.tokenizer.decode(generations['sequences'][0], skip_special_tokens=True),
  }

def get_final_output_string(model, inputs, generations, answer_prefix: str) -> str:
  generated_tokens = generations.sequences[0][inputs['input_ids'].shape[1]:]
  output_text = model.tokenizer.decode(
      generated_tokens, skip_special_tokens=True
  ).strip()
  return answer_prefix + output_text

def parse_numbered_reasons(generation_string: str, marker: str, tokenizer) -> dict[str, torch.Tensor]:
  if marker not in generation_string:
    return {}

  reasons_text = generation_string.split(marker)[-1]

  if 'None' in reasons_text or 'none' in reasons_text.lower():
    return {}

  reason_pattern = r'(\d+)\.\s*([^0-9]+?)(?=\s*\d+\.|$)'
  matches = re.findall(reason_pattern, reasons_text, re.DOTALL)

  reasons = {}
  for _, reason_text in matches:
    cleaned_reason = reason_text.strip().replace('\n', ' ')
    if cleaned_reason:
      reasons[cleaned_reason] = tokenizer.encode(cleaned_reason, return_tensors='pt')[0][1:-1]

  return reasons

def get_reason_entropies(logits, scores, reasons: dict[str, torch.Tensor], input_tokens: torch.Tensor, output_tokens: torch.Tensor) -> tuple[list[list[float]], list[list[float]]]:
  """Calculate token-wise predictive entropies for each reason.

    Implements: U(rⱼ, x) = log p(zᵢ | r<ᵢ, x)
    where x is input tokens and we calculate separate entropies for each reason rⱼ.

    Args:
      logits: Model logits for each decision.
      scores: Model scores for each decision.
      reasons: List of reasons to calculate entropies for.
      associated_token_ranges: List of tuples of (start_token_index, end_token_index) for each reason.
      output_tokens: The actual output token sequence.
    Returns:
      Tuple of (reason_logits_entropies, reason_scores_entropies) as lists of tensors,
      where each element corresponds to one reason.
    """
  logits_entropies = []
  scores_entropies = []

  # we can chop up to MAX_WIGGLE_ROOM tokens from the start of the reason to still be considered a match
  # yes this is sketchy i will think of a better way later :)
  MAX_WIGGLE_ROOM = 3

  for reason, tokens in reasons.items():
    start_idx = np.inf
    wiggle_room = 0
    while start_idx == np.inf and wiggle_room < MAX_WIGGLE_ROOM:
      for i in range(len(output_tokens)-len(tokens)+1):
        if torch.equal(output_tokens[i:i+len(tokens[wiggle_room:-1])], tokens[wiggle_room:-1]):
          start_idx = i
          end_idx = i + len(tokens[wiggle_room:-1])
          tokens_shifted = tokens[wiggle_room:-1]
          break
      wiggle_room += 1

    if start_idx == np.inf:
      raise ValueError(f"Reason {reason} not found in output tokens")

    input_length = input_tokens.shape[1]
    reason_logits = logits[0, start_idx - input_length:end_idx - input_length]
    reason_scores = scores[0, start_idx - input_length:end_idx - input_length]

    if len(tokens) > 1:
      # entropy for each reason
      logits_entropy = torch.nn.CrossEntropyLoss(reduction='none')(reason_logits, tokens_shifted)
      scores_entropy = torch.nn.CrossEntropyLoss(reduction='none')(reason_scores, tokens_shifted)
      logits_entropies.append(logits_entropy.tolist())
      scores_entropies.append(scores_entropy.tolist())
    else:
      logits_entropies.append([])
      scores_entropies.append([])

  return logits_entropies, scores_entropies


def save_haf_metrics(metrics_dict: dict, output_file_path: str) -> dict:
  """Save HAF metrics to a pickle file."""
  with open(output_file_path, 'wb') as f:
    pickle.dump(metrics_dict, f)
  return metrics_dict
