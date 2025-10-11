import torch
from sentence_transformers import CrossEncoder
from concordia.haf_integration.haf_utils import pkl_to_dict
from concordia.haf_integration.haf_common_metrics import (
    diversity,
    relevance_scores,
    confidence_scores,
)


class UpholdReasonsStageMetrics:
  """
  Calculates metrics for the uphold-reasons stage of HAF evaluation (UII and UEI).
  """
  def __init__(
      self,
      uphold_reasons_file_path: str,
      justify_stage_file_path: str,
      similarity_model: CrossEncoder,
      tokenizer,
  ):
    self.uphold_reasons_file_path = uphold_reasons_file_path
    self.justify_stage_file_path = justify_stage_file_path
    self.similarity_model = similarity_model
    self.tokenizer = tokenizer
    self.w_c = 0.5
    self.w_g = 0.5

  def extract_uphold_reasons_stage_data(self, uphold_reasons_stage_file_path: str) -> tuple:
    data = pkl_to_dict(uphold_reasons_stage_file_path)

    return (
        data['agent_name'],
        data['input_text'],
        data['decision'],
        data['original_reasons'],
        data['is_complete'],
        data['additional_reasons'],
        data['output_tokens'],
        data['logits'],
        data['scores'],
    )

  def extract_justify_stage_data(self, justify_stage_file_path: str) -> tuple:
    data = pkl_to_dict(justify_stage_file_path)

    return (
        data['agent_name'],
        data['input_text'],
        data['output_tokens'],
        data['reasons'],
        data['reason_logits_entropies'],
        data['reason_scores_entropies'],
    )

  def _calculate_completeness_confidence(
      self, logits, scores, output_tokens: torch.Tensor
  ) -> tuple[float, float]:
    """Calculate confidence in the completeness decision from logits and scores."""
    logits_probs = torch.nn.functional.softmax(logits[0], dim=-1)
    scores_probs = torch.nn.functional.softmax(scores[0], dim=-1)

    logits_confidence = torch.mean(torch.max(logits_probs, dim=-1)[0]).item()
    scores_confidence = torch.mean(torch.max(scores_probs, dim=-1)[0]).item()

    return logits_confidence, scores_confidence

  def _calculate_diversity(
      self,
      additional_reasons: dict[str, torch.Tensor],
      original_reasons_with_tokens: dict[str, torch.Tensor],
      original_confidence_scores: list[float],
  ) -> float:
    """Calculate total diversity across all additional reasons."""
    if not additional_reasons:
      return 0.0

    total_diversity = 0.0
    for additional_reason in additional_reasons.keys():
      div_score = diversity(
          additional_reason,
          original_reasons_with_tokens,
          self.similarity_model,
          original_confidence_scores,
      )
      total_diversity += div_score

    return total_diversity

  def unused_internal_information(self, is_complete: bool, confidence: float, diversity: float) -> float:
    """
    Calculate Unused Internal Information (UII) metric.

    UII = w_c * C(Y^(UR), x^(UR)) * div(r_j^(UR), R^(J)) + w_g * div(r_j^(UR), R^(J))

    Where:
    - C is confidence in the completeness decision
    - div is diversity between additional and original reasons
    - w_c and w_g are weights
    """

    confidence_score = self.w_c * confidence * (0 if is_complete else 1)
    diversity_score = self.w_g * diversity if diversity > 0 else 0

    return confidence_score + diversity_score

  def compute_uphold_reasons_stage_metrics(self) -> dict:
    """Compute all HAF metrics from the uphold-reasons stage."""

    (
        agent_name,
        input_text,
        decision,
        original_reasons,
        is_complete,
        additional_reasons,
        output_tokens,
        logits,
        scores,
    ) = self.extract_uphold_reasons_stage_data(self.uphold_reasons_file_path)

    (
        _,
        _,
        justify_output_tokens,
        original_reasons_with_tokens,
        reason_logits_entropies,
        reason_scores_entropies,
    ) = self.extract_justify_stage_data(self.justify_stage_file_path)

    logits_confidence, scores_confidence = self._calculate_completeness_confidence(
        logits, scores, output_tokens
    )

    relevance_scores_list = relevance_scores(
        original_reasons_with_tokens, self.similarity_model, self.tokenizer
    )
    original_confidence_scores = confidence_scores(
        relevance_scores_list, reason_scores_entropies, self.tokenizer
    )

    diversity_score = self._calculate_diversity(
        additional_reasons,
        original_reasons_with_tokens,
        original_confidence_scores,
    )

    uii = self.unused_internal_information(
        is_complete, scores_confidence, diversity_score
    )

    metrics_dict = {
        'UII': uii,
        'is_complete': is_complete,
        'completeness_confidence': scores_confidence,
        'diversity': diversity_score,
        'num_additional_reasons': len(additional_reasons),
        'num_original_reasons': len(original_reasons),
    }

    return metrics_dict


if __name__ == '__main__':
  from sentence_transformers import CrossEncoder
  from transformers import AutoTokenizer

  uphold_reasons_file_path = (
      'concordia/haf_integration/uphold_reasons_stage/data/uphold_reasons_stage_data_1649-01-29'
      ' 13:00:00_King Charles I.pkl'
  )
  justify_stage_file_path = (
      'concordia/haf_integration/justify_stage/data/justify_stage_data_1649-01-29'
      ' 13:00:00_King Charles I.pkl'
  )
  similarity_model = CrossEncoder('sentence-transformers/all-MiniLM-L6-v2')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

  metrics = UpholdReasonsStageMetrics(
      uphold_reasons_file_path,
      justify_stage_file_path,
      similarity_model,
      tokenizer,
  )
  print(metrics.compute_uphold_reasons_stage_metrics())
