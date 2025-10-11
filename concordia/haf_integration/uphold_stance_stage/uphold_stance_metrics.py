import torch
from concordia.haf_integration.haf_utils import pkl_to_dict


class UpholdStanceStageMetrics:
  """
  Calculates metrics for the uphold-stance stage of HAF evaluation (RS).
  """
  def __init__(self, output_file_path: str):
    self.output_file_path = output_file_path

  def extract_uphold_stance_stage_data(self, uphold_stance_stage_file_path: str) -> tuple:
    data = pkl_to_dict(uphold_stance_stage_file_path)

    return (
        data['agent_name'],
        data['input_text'],
        data['decision'],
        data['original_reasons'],
        data['reason_evaluations'],
    )

  def reason_sufficiency(self, reason_evaluations: list[dict]) -> float:
    """
    Calculate Reason Sufficiency (RS) metric.

    RS = w_S * C(Y^(r_j), x^(r_j))

    Where:
    - C is confidence in the sufficiency decision Y^(r_j)
    - w_S is weight for confidence (1.0 if sufficient, 0.0 otherwise)
    """
    if not reason_evaluations:
      return 0.0

    rs_scores = []
    for eval_data in reason_evaluations:
      is_sufficient = eval_data['is_sufficient']
      logits_confidence, scores_confidence = self._calculate_sufficiency_confidence(
          eval_data['logits'],
          eval_data['scores'],
          eval_data['output_tokens']
      )

      w_s = 1.0 if is_sufficient else 0.0
      rs = w_s * scores_confidence
      rs_scores.append(rs)

    return sum(rs_scores) / len(rs_scores) if rs_scores else 0.0

  def _calculate_sufficiency_confidence(self, logits, scores, output_tokens: torch.Tensor) -> tuple[float, float]:
    """Calculate confidence in the sufficiency decision."""
    first_token_logits = logits[0, 0]
    first_token_scores = scores[0, 0]

    logits_probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
    scores_probs = torch.nn.functional.softmax(first_token_scores, dim=-1)

    actual_token = output_tokens[0]
    logits_confidence = logits_probs[actual_token].item()
    scores_confidence = scores_probs[actual_token].item()

    return logits_confidence, scores_confidence

  def compute_uphold_stance_stage_metrics(self) -> dict:
    """Compute all HAF metrics from the uphold-stance stage."""

    (agent_name, input_text, decision, original_reasons,
     reason_evaluations) = self.extract_uphold_stance_stage_data(self.output_file_path)

    rs = self.reason_sufficiency(reason_evaluations)

    num_sufficient = sum(1 for e in reason_evaluations if e['is_sufficient'])

    avg_confidence = 0.0
    if reason_evaluations:
      confidences = [
          self._calculate_sufficiency_confidence(e['logits'], e['scores'], e['output_tokens'])[1]
          for e in reason_evaluations
      ]
      avg_confidence = sum(confidences) / len(confidences)

    metrics_dict = {
        'RS': rs,
        'num_sufficient_reasons': num_sufficient,
        'num_reasons_evaluated': len(reason_evaluations),
        'avg_confidence': avg_confidence,
    }

    return metrics_dict


if __name__ == '__main__':
  uphold_stance_stage_file_path = (
      'concordia/haf_integration/uphold_stance_stage/data/uphold_stance_stage_data_00:00:00_King'
      ' Charles I.pkl'
  )
  metrics = UpholdStanceStageMetrics(uphold_stance_stage_file_path)
  print(metrics.compute_uphold_stance_stage_metrics())
