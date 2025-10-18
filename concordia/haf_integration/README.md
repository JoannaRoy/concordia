# Human-Aligned Faithfulness (HAF) for Concordia

**Understanding agent decision-making in generative social simulations via a Human-Aligned Faithfulness criterion**

## Background

This project is an extension of the [Human-Aligned Faithfulness (HAF) framework](https://arxiv.org/abs/2506.19113) for evaluating LLM toxicity explanations. Building on that initial work, this project explores how HAF can serve as a mechanism for evaluating agent reasoning abilities (I am specifically interested in **public policy contexts**). This work tries to use the HAF criterion to evaluate agent reasoning complex policy scenarios. This could, for example, allow policymakers to use libraries like concordia to 'play out' potential policies and better predict their effectiveness before recommending them in the real world.

## Project Overview

This project integrates the Human-Aligned Faithfulness (HAF) criterion into [Concordia](https://github.com/google-deepmind/concordia), a generative social simulation platform developed by Google DeepMind. HAF provides a framework for evaluating the reliability and quality of agent decision-making in social simulations.

## Overview

Generative social simulations with LLM agents are increasingly used as tools for informing policy decisions and intervention recommendations. However, for simulations to serve as reliable support for decision-makers, we need rigorous methods to evaluate how much weight should be placed in the decisions and recommendations made by generative agents.

**HAF addresses this challenge** by providing a multi-dimensional criterion for assessing agent decision-making relative to a rational human under ideal conditions (Rational Human under Ideal conditions - RHI). This enables researchers to understand not just *what* decisions agents make, but *how well-justified* those decisions are.

## The HAF Criterion

The HAF criterion evaluates agent decisions across **five key axes**:

### 1. **Relevance (REL)**
The reasons provided by the agent should imply something about the likelihood of the conclusion.

**Metrics:**
- **Stance-of-Support (SoS)**: Measures strength of support and diversity in supporting reasons
  ```
  SoS = 1/|R^(J)| * Σ[w_C^(J) * C(r_j^(J), x^(J)) + w_D^(J) * div(r_j^(J), d_rm)]
  ```
  where `w_C^(J) = 0.8` and `w_D^(J) = 0.5`

- **Diversity-in-Support (DiS)**: Quantifies diversity in reasons using Unused Internal Information (UII) and Unused External Information (UEI)

### 2. **Internal Reliance (INT)**
Ideal explanations should encode all possible information about the world to arrive at the conclusion.

**Metric: Unused Internal Information (UII)**
```
UII = 1/|R^(UR)| * Σ[w_C^(UR) * C(r_i^(UR), x^(UR)) + w_D^(UR) * div(r_i^(UR), R^(J))]
```
where `w_C^(UR) = w_D^(UR) = 0.5`

*Lower values are more desirable* (indicates fewer unused internal reasons).

### 3. **External Reliance (EXT)**
Ideal explanations should encode all possible information about the world to arrive at the conclusion.

**Metric: Unused External Information (UEI)**

Defined analogously to UII (not shown for brevity).

*Lower values are more desirable* (indicates fewer unused external reasons).

### 4. **Sufficiency (SUF)**
Assesses whether each provided reason is sufficient (or necessary) to justify the stance.

**Metric: Reason Sufficiency (RS)**
```
RS = w_S * C * (Ȳ^(J), x^(J,s)) * (1 - I_S(S^(J)))
```
where `I_S(S^(J))` determines whether the jth reason is sufficient on its own, and `w_S = 0.5`.

*Higher values are more desirable*.

### 5. **Necessity (NEC)**
Assesses whether each provided reason is sufficient or necessary to justify the stance.

**Metric: Reason Necessity (RN)**
```
RN = w_N * C * (Ȳ^(J,R), x^(J,R)) * (1 - I_N(S^(J,R)))
```
where `w_N = 0.5`.

*Higher values are more desirable*.

## Three-Stage Pipeline

The HAF evaluation pipeline is divided into three stages, each assessing different axes of the criterion:

### Stage 1: Justify Stage
**Evaluates:** Relevance (REL)

Agents are prompted to provide decisions and associated reasonings in the format:
```
DECISION: [Your decision]
REASON(S):
1. [reason text]
2. [reason text]
...
```

The stage calculates:
- Confidence scores for each reason based on token-wise uncertainty
- Semantic relevance scores
- Token-wise predictive entropies

### Stage 2: Uphold Reasons Stage
**Evaluates:** Internal Reliance (INT) and External Reliance (EXT)

The agent is asked whether the reasons from Stage 1 are complete or if additional reasons are needed. This evaluates:
- Quality of the original reasoning
- Completeness of the reasoning set
- Unused internal/external information

### Stage 3: Uphold Stance Stage
**Evaluates:** Sufficiency (SUF) and Necessity (NEC)

Each individual reason from Stage 1 is evaluated to determine:
- Whether it is individually sufficient to justify the decision
- Whether it is necessary for the decision
- The agent's understanding of connections between reasons and decisions

## Integration with Concordia

The HAF pipeline integrates seamlessly with Concordia simulations. During simulation, each time an agent or Game Master makes a decision, the HAF pipeline "interrupts" the simulation to evaluate the decision quality.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Concordia Simulation                        │
│                                                                 │
│  ┌──────────┐         Decision          ┌──────────────┐       │
│  │  Agent/  │───────► Required ────────►│  HAF         │       │
│  │   GM     │                           │  Evaluation  │       │
│  └──────────┘         ◄──────────────── └──────────────┘       │
│                   Decision Passed Back                          │
│                   to Simulation Context                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **`HAFWrapper`**: Orchestrates all three HAF stages consecutively
2. **`HAFJustifyStage`**: Generates decisions with reasons (Stage 1)
3. **`HAFUpholdReasonStage`**: Evaluates reason completeness (Stage 2)
4. **`HAFUpholdStanceStage`**: Evaluates individual reason sufficiency (Stage 3)
5. **`haf_common_metrics.py`**: Implements the mathematical metrics for all axes
6. **`haf_utils.py`**: Utility functions for parsing, tokenization, and entropy calculation

## Installation

The HAF integration is part of the Concordia repository. To use it:

```bash
git clone https://github.com/google-deepmind/concordia.git
cd concordia
pip install -e .
```

Ensure you have the required dependencies:
- `sentence_transformers` (for semantic similarity)
- `torch` (for model operations)
- `transformers` (for tokenization)

## Usage

### Basic Usage

```python
from concordia.language_model.huggingface_model import HuggingFaceLanguageModel
from concordia.haf_integration.haf_wrapper import HAFWrapper

model = HuggingFaceLanguageModel(model_name="your-model-name")

haf = HAFWrapper(
    agent_name="King Charles I",
    current_time_str="00:00:00",
    action_spec=your_action_spec,
    general_guidance_text="Context for the agent...",
    model=model
)

result = haf.run()
```

### Using Individual Stages

```python
from concordia.haf_integration.justify_stage.justify import HAFJustifyStage
from concordia.haf_integration.uphold_reasons_stage.uphold_reasons import HAFUpholdReasonStage
from concordia.haf_integration.uphold_stance_stage.uphold_stance import HAFUpholdStanceStage

justify_stage = HAFJustifyStage(
    agent_name="Agent Name",
    current_time_str="timestamp",
    action_spec=action_spec,
    general_guidance_text="context",
    model=model
)
justify_result = justify_stage.run()
```

### Output Data

Each stage saves processed data as pickle files:
- `justify_stage_data_{timestamp}_{agent_name}.pkl`
- `uphold_reasons_stage_data_{timestamp}_{agent_name}.pkl`
- `uphold_stance_stage_data_{timestamp}_{agent_name}.pkl`

The data includes:
- Agent decisions and reasons
- Token-wise entropies
- Logits and scores
- Prompts and outputs
- Evaluation results (completeness, sufficiency, etc.)

## Metrics Calculation

### Confidence Scores

Confidence for each reason is calculated based on token-wise uncertainty:

```python
from concordia.haf_integration.haf_common_metrics import confidence_scores

confidences = confidence_scores(relevance_scores, reason_entropies)
```

### Diversity

Diversity measures how different an additional reason is from the original set:

```python
from concordia.haf_integration.haf_common_metrics import diversity

div_score = diversity(
    additional_reason,
    original_reasons,
    similarity_model,
    confidence_scores_list
)
```

### Relevance

Relevance scores are calculated at the token level within each reason:

```python
from concordia.haf_integration.haf_common_metrics import relevance_scores

rel_scores = relevance_scores(reasons, similarity_model, tokenizer)
```

## Example: Early Results

Early evaluation on simple simulations shows:
- **Stage I: Justify** - SoS: Varies by scenario and model
- **Stage II: Uphold Reasons** - UII: Quantifies unused internal information
- **Stage III: Uphold Stance** - RS, RN: Evaluates individual reason quality

The metrics allow comparison between different models and decision types to understand where agents are more or less sure of themselves.

## Technical Details

### Model Requirements

HAF requires a HuggingFace-compatible language model that:
- Supports generation with logits/scores output
- Has a tokenizer with proper encoding/decoding
- Can generate structured text following specific formats

### Configuration

The SAFTE (Semantic Analysis for Faithfulness and Trust Evaluation) configuration is defined in `haf_utils.py`:

```python
SAFTE_CONFIG = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 1.0,
    'max_new_tokens': 100,
    'return_dict_in_generate': True,
    'output_scores': True,
    'output_logits': True,
}
```

## Future Work

This project is part of ongoing research into:
- Running tests on more samples and diverse policy scenarios
- Comparing metrics between models and decision types
- Improving evaluation of agent decision-making in many-agent scenarios
- Exploring how HAF scores correlate with real-world decision quality

## References

1. **Concordia**: Park, J. S., Popov, B., Lerer, A., Mukobi, G., Vezhnevets, A. S., Duéñez-Guzmán, E. A., ... & Summerfield, C. (2024). "Generative agents: Interactive simulacra of human behavior." arXiv:2304.03442.

2. **K. E. Atkinson, J. Roy, J. Morack, S. Guha.** "Human-Aligned Faithfulness in Trusty Explanations (of LLMs)." (In preparation).

3. **K. E. Atkinson, J. Maillard, D. Roy, I. Laponogov, A. Vezhnevets, E. A. Duéñez-Guzmán, & S. Guha.** "Decision-making actions grounded in physical, social, or digital space using Concordia." arXiv preprint arXiv:2312.03664.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{roy2025haf,
  title={Are you sure about that? Understanding agent decision-making in generative social simulations via a Human-Aligned Faithfulness criterion},
  author={Roy, Joanna and others},
  year={2025},
  howpublished={Cooperative AI Foundation Summer School Poster Presentation}
}
```

## Authors

- Joanna Roy (University of Toronto)
- Ramravindr K. Meenhal
- Syed Tahsique Ahmed
- Shion Guha

## License

This project is part of the Concordia framework and follows the same Apache 2.0 license.

## Acknowledgments

This work was presented at the Cooperative AI Foundation Summer School Poster Presentation (July 5th to 13th, 2025 | Marlow, United Kingdom).

Special thanks to the Concordia team at Google DeepMind for developing the foundational simulation platform.
