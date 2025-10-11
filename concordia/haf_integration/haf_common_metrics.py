import pickle
import torch
from sentence_transformers import CrossEncoder
from transformers.agents.llm_engine import AutoTokenizer


def input_reasons_similarities(
    input_text: str,
    reasons: dict[str, torch.Tensor],
    similarity_model: CrossEncoder,
) -> tuple[list[float], list[list[float]]]:
  """Calculate similarity scores between input text and reasons, and between reasons.

  computes g(x,y) = semantic similarity between x and y (using a similarity model)

  Args:
    input_text: The input context text.
    reasons: List of reason strings to analyze.

  Returns:
    Tuple containing:
      - List of similarity scores between input text and each reason
      - List of similarity scores between each reason and every other reason
  """
  input_reason_pairs = [(input_text, reason) for reason in reasons.keys()]
  if input_reason_pairs:
    scores = similarity_model.predict(input_reason_pairs)
    with_input = [float(s) for s in scores]
  else:
    with_input = []

  between_reasons = []
  for i, reason_i in enumerate(reasons):
    similarity_scores_i = []
    for j, reason_j in enumerate(reasons):
      if i != j:
        similarity_scores_i.append(similarity_model.predict([reason_i, reason_j]))
    between_reasons.append(similarity_scores_i)

  return with_input, between_reasons


def relevance_per_reason(reason: str, tokens: torch.Tensor, similarity_model: CrossEncoder, tokenizer: AutoTokenizer) -> list[float]:
  """Calculate semantic relevance scores for each token in a reason.

  Implements: S(z_i, r_j) = 1 - |g(r_j, r_j \ {z_i})| (where g() is a semantic similarity function)
  where z_i are tokens, r_j is the reason, and g() is a semantic similarity function.

  Args:
    reason: The reason text string
    tokens: Token encoding for this reason

  Returns:
    List of semantic relevance scores for each token in the reason
  """
  relevance_scores = []

  for i, token in enumerate(tokens):
    # Decode the individual token
    decoded_token = tokenizer.decode([token], skip_special_tokens=True)

    # Create reason without this token (r_j \ {z_i})
    reason_without_token = reason.replace(decoded_token, '', 1)  # Remove only first occurrence

    # Handle edge case where token removal results in empty or identical string
    if reason_without_token.strip() == reason.strip() or not reason_without_token.strip():
      # If token wasn't found or removal results in empty string, assign low relevance
      relevance_scores.append(0.0)
    else:
      # Calculate similarity g(r_j, r_j \ {z_i})
      similarity = similarity_model.predict([[reason, reason_without_token]])[0]

      # S(z_i, r_j) = 1 - |g(r_j, r_j \ {z_i})|
      semantic_relevance = 1 - abs(float(similarity))
      relevance_scores.append(semantic_relevance)

  # Normalize the relevance scores: Ŝ(z_i, r_j) = S(z_i, r_j) / Σₖ S(z_k, r_j)
  total_relevance = sum(relevance_scores)
  if total_relevance > 0:
    relevance_scores_normalized = [r / total_relevance for r in relevance_scores]
  else:
    # If all scores are 0, distribute equally
    relevance_scores_normalized = [1.0 / len(relevance_scores) if relevance_scores else 0.0] * len(relevance_scores)

  return relevance_scores_normalized


def relevance_scores(reasons: dict[str, torch.Tensor], similarity_model: CrossEncoder, tokenizer: AutoTokenizer) -> list[list[float]]:
  """Calculate relevance scores for each reason based on its contribution to overall meaning.

  Args:
    reasons: List of reason text strings

  Returns:
    List of relevance score lists, one list per reason
  """
  relevance_scores = []
  for reason, tokens in reasons.items():
    relevance_scores.append(relevance_per_reason(reason, tokens, similarity_model, tokenizer))
  return relevance_scores

def confidence_per_reason(reason_relevance_scores: list[float], reason_entropy: list[float]) -> float:
  """Compute confidence score for each reason based on token-wise relevances and entropies.

  Implements: U(r_j, x) = Σᵢ₌₁ᴺʲ [-log p(z_i | r_{<i}, x) * S̃(z_i, r_j)]
  where we now have separate entropies for each reason r_j.

  Args:
    reason_relevance_scores: List of relevance scores for tokens in this reason.
    reason_entropy: List of entropy values for tokens in this reason.

  Returns:
    Confidence score for this reason.
  """
  # reason may have had some tokens chopped off during the justify stage entropy calculation, so we need to account for that
  wiggle_room = abs(len(reason_entropy) - len(reason_relevance_scores))

  # Calculate token-wise uncertainty: -log p(z_i | r_{<i}, x) * S̃(z_i, r_j)
  token_uncertainty_sum = sum([
      reason_entropy[i] * reason_relevance_scores[i+wiggle_room]
      for i in range(len(reason_entropy))
  ])

  return torch.exp(-torch.tensor(token_uncertainty_sum)).item()


def confidence_scores(
    relevance_scores: list[list[float]], reason_entropies: list[list[float]]
) -> list[float]:
  """Compute confidence score for each reason based on token-wise relevances and entropies.

  Args:
    relevance_scores: List of relevance score lists, one for each reason.
    reason_entropies: List of entropy lists, one for each reason.

  Returns:
    List of confidence scores, one for each reason.
  """
  confidence_scores = []
  for reason_relevance_scores, reason_entropy in zip(relevance_scores, reason_entropies):
    confidence_scores.append(
        confidence_per_reason(reason_relevance_scores, reason_entropy)
    )
  return confidence_scores


def diversity(
    additional_reason: str,
    reasons: dict[str, torch.Tensor],
    similarity_model: CrossEncoder,
    confidence_scores_list: list[float]
) -> float:
    """Calculate diversity of one additional reason against all original reasons.

    Implements: div(r_i^(UR), R^(J)) = (|R^(J)|/Σ_k) [h(r_i^(UR), r_k^(J)) · C(r_k^(J), x^(J))] / (Σ_k C(r_k^(J), x^(J)))
    where h(x,y) = 1 - g(x,y) and g(x,y) is semantic similarity

    Args:
        additional_reason: Single additional reason from uphold-reasons stage (r_i^(UR))
        reasons: Original reasons from justify stage (R^(J))
        similarity_model: CrossEncoder model for computing semantic similarity
        confidence_scores_list: Confidence scores for each original reason C(r_k^(J), x^(J))

    Returns:
        Diversity score for the additional reason
    """
    if not additional_reason or not reasons:
        return 0.0

    original_reasons_list = list(reasons.keys())
    num_original_reasons = len(original_reasons_list)

    if len(confidence_scores_list) != num_original_reasons:
        raise ValueError(f"Number of confidence scores ({len(confidence_scores_list)}) must match number of original reasons ({num_original_reasons})")

    sum_confidences = sum(confidence_scores_list)
    if sum_confidences == 0:
        return 0.0

    weighted_dissimilarity_sum = 0.0

    for k, (original_reason, confidence) in enumerate(zip(original_reasons_list, confidence_scores_list)):
        similarity = similarity_model.predict([[additional_reason, original_reason]])[0]
        dissimilarity = 1.0 - float(similarity)

        weighted_dissimilarity_sum += dissimilarity * confidence

    diversity_score = (num_original_reasons * weighted_dissimilarity_sum) / sum_confidences

    return diversity_score
