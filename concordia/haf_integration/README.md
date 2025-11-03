# Human-Aligned Faithfulness (HAF) for Concordia

**Understanding agent decision-making in generative social simulations via a Human-Aligned Faithfulness criterion**

## Background

This project is an extension of a paper I worked on recently ([Human-Aligned Faithfulness (HAF)](https://arxiv.org/abs/2506.19113) for evaluating LLM toxicity explanations). Rather than looking at HAF in the context of toxic content, this project looks at how HAF can be used to evaluate agent reasoning abilities in multi-agent simulations. My hope is to use the HAF criterion to evaluate agent reasoning complex policy scenarios. This could, for example, allow policymakers to use libraries like concordia to 'play out' potential policies and better predict their effectiveness before recommending them in the real world.

Below is some info from a (early work) poster I presented a couple months ago at the Cooperative AI Foundation Summer School. The project is still very much a work-in-progress, so I'm grateful for any feedback or ideas (as issues on here or email to joannaroy6@gmail.com).

## Overview

Generative social simulations with LLM agents could potentially be used as tools for informing policy decisions and intervention recommendations. However, for simulations to serve as reliable support for decision-makers, we need rigorous methods to evaluate how much weight should be placed in the decisions and recommendations made by generative agents.

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

### 3. **Sufficiency (SUF)**
Assesses whether each provided reason is sufficient (or necessary) to justify the stance.

**Metric: Reason Sufficiency (RS)**
```
RS = w_S * C * (Ȳ^(J), x^(J,s)) * (1 - I_S(S^(J)))
```
where `I_S(S^(J))` determines whether the jth reason is sufficient on its own, and `w_S = 0.5`.

*Higher values are more desirable*.

### 4. **Necessity (NEC)**
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
**Evaluates:** Internal Reliance (INT)

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

## Future Work

This project is part of ongoing research into:
- Running tests on more samples and diverse policy scenarios
- Comparing metrics between models and decision types
- Improving evaluation of agent decision-making in many-agent scenarios
- Exploring how HAF scores correlate with real-world decision quality

## License

This project is part of the Concordia framework and follows the same Apache 2.0 license.
