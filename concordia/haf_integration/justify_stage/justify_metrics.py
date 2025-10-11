import torch
from concordia.haf_integration.haf_utils import pkl_to_dict, save_haf_metrics
from concordia.haf_integration.haf_common_metrics import input_reasons_similarities, relevance_scores, confidence_scores
from sentence_transformers import CrossEncoder
from transformers.agents.llm_engine import AutoTokenizer


class JustifyStageMetrics:
  """
  Calculates metrics for the justify stage of HAF evaluation (SoS and DiS).
  """
  def __init__(self, tokenizer: AutoTokenizer, similarity_model: CrossEncoder, output_file_path: str):
    self.tokenizer = tokenizer
    self.similarity_model = similarity_model
    self.output_file_path = output_file_path
    self.w_c = 0.8
    self.w_g = 0.2

  @staticmethod
  def extract_justify_stage_data(justify_stage_file_path: str) -> tuple[
      str,
      str,
      list[int],
      dict[str, torch.Tensor],
      list[list[float]],
      list[list[float]],
  ]:
    justify_stage_data = pkl_to_dict(justify_stage_file_path)

    return (
          justify_stage_data['agent_name'],
          justify_stage_data['input_text'],
          justify_stage_data['output_tokens'],
          justify_stage_data['reasons'],
          justify_stage_data['reason_logits_entropies'],
          justify_stage_data['reason_scores_entropies']
      )

  def strength_of_support(self, input_reason_similarities: list[float], confidence_scores: list[float]) -> float:
    """
    Calculate the strength of support for the decision using the SoS equation.

    Args:
      input_reason_similarities: List of similarity scores between input text and reasons (for a given decision).
      confidence_score: Confidence score for the decision.

    Returns:
      Strength of support score for the decision.
    """

    num_terms = len(input_reason_similarities) # number of reasons given for the decision

    total_support = sum(
        self.w_c * confidence_scores[i]
        + self.w_g * input_reason_similarities[i]
        for i in range(num_terms)
    )

    sos = total_support / num_terms if num_terms > 0 else 0.0
    return sos

  def diversity_in_supports(self, between_reason_similarities: list[list[float]], confidence_scores: list[float]) -> list[float]:
    """
    Calculate the diversity in support for the decision using the DiS equation.
    """
    num_reasons = len(between_reason_similarities)
    dis = []
    for i, reason_reason_similarities in enumerate(between_reason_similarities):
      h = 0
      for j, g_r_i in enumerate(reason_reason_similarities):
        if i != j:
          h += (1 - g_r_i) * confidence_scores[j]

      dis.append(h / num_reasons * (num_reasons - 1))

    return dis

  def compute_justify_stage_metrics(self) -> dict:
    """Compute all HAF metrics from the justify stage decision entry."""

    _, input_text, _, reasons, reason_logits_entropies, _ = (
        self.extract_justify_stage_data(self.output_file_path)
    )

    # input-reason similarities for each reason; reason-reason similarities for each reason relative to all other reasons
    similarity_data = input_reasons_similarities(input_text, reasons, self.similarity_model)
    input_reason_similarities: list[float] = similarity_data[0]
    between_reason_similarities: list[list[float]] = similarity_data[1]

    # list of relevance scores for each reason
    relevance_scores_list: list[list[float]] = relevance_scores(reasons, self.similarity_model, self.tokenizer)

    # list of confidence scores for each reason
    confidence_scores_list: list[float] = confidence_scores(
        relevance_scores_list, reason_logits_entropies
    )  # TODO confirm its not score_entropies

    # strength of support for the decision (based on confidence and similarity of each reason to the input text)
    sos: float = self.strength_of_support(input_reason_similarities, confidence_scores_list)

    # diversity in support for the decision (based on confidence of each reason relative to all other reasons)
    dos: list[float] = self.diversity_in_supports(between_reason_similarities, confidence_scores_list)

    metrics_dict = {
        'input_reasons_similarities': input_reason_similarities,
        'between_reasons_similarities': between_reason_similarities,
        'relevance_scores': relevance_scores_list,
        'confidence_scores_per_reason': confidence_scores_list,
        'strength_of_support': sos,
        'diversity_in_supports': dos,
    }

    save_haf_metrics(metrics_dict, self.output_file_path)

    return metrics_dict

if __name__ == '__main__':
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
  similarity_model = CrossEncoder('sentence-transformers/all-MiniLM-L6-v2')
  justify_stage_file_path = (
      'concordia/haf_integration/justify_stage/data/justify_stage_data_00:00:00_King'
      ' Charles I.pkl'
  )
  metrics = JustifyStageMetrics(tokenizer, similarity_model, justify_stage_file_path)
  metrics.compute_justify_stage_metrics()
