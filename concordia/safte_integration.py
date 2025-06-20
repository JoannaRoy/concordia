import torch
import json
from concordia.language_model.huggingface_model import HuggingFaceLanguageModel

SAFTE_CONFIG = {
    'pad_token_id': None,
    'eos_token_id': None,
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 1.0,
    'max_new_tokens': 100,
    'return_dict_in_generate': True,
    'output_scores': True,
    'output_logits': True,
}


class SAFTEJustifyStage:

  def __init__(
      self,
      agent_name: str,
      current_time_str: str,
      call_to_action: str,
      scenario_context: str,
      model: HuggingFaceLanguageModel,
      file_path: str = 'justify_stage.json',
      **config_overrides,
  ):
    # identifiers
    self.agent_name = agent_name
    self.current_time_str = current_time_str

    # stage inputs:
    self.call_to_action = call_to_action
    self.scenario_context = scenario_context
    self.model = model
    self.file_path = file_path
    self.config = {**SAFTE_CONFIG, **config_overrides}

    # intermediate outputs:
    self.prompt = self.format_prompt(scenario_context)
    self.input_tokens = None
    self.output_tokens = None
    self.logits = None
    self.scores = None

    # stage outputs:
    self.processed_logits = None
    self.processed_scores = None
    self.generated_text = None

  def format_prompt(self, scenario_context: str):
    return [
        self.model.tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': self.call_to_action},
                {
                    'role': 'user',
                    'content': '\nTEXT: \n' + scenario_context.lstrip(),
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    ]

  @staticmethod
  def get_entropies(input_tokens, output_tokens, logits, scores):
    # token-wise predictive entropies
    processed_logits = []
    processed_scores = []
    for sample_ix in range(len(input_tokens)):
      sample_input_len = len(input_tokens[sample_ix])
      target_ids = output_tokens[sample_ix].clone()[sample_input_len:]
      token_wise_entropy_logits = torch.nn.CrossEntropyLoss(reduction='none')(
          logits[sample_ix], target_ids
      )
      token_wise_entropy_scores = torch.nn.CrossEntropyLoss(reduction='none')(
          scores[sample_ix], target_ids
      )
      processed_logits.append(token_wise_entropy_logits)
      processed_scores.append(token_wise_entropy_scores)

    return torch.stack(processed_logits), torch.stack(processed_scores)

  def save_to_json(self):
    data = {
        'agent_name': self.agent_name,
        'scenario_context': self.scenario_context,
        'input_tokens': self.input_tokens.tolist(),
        'output_tokens': self.output_tokens.tolist(),
        'logits': self.logits.tolist(),
        'scores': self.scores.tolist(),
        'processed_logits': self.processed_logits.tolist(),
        'processed_scores': self.processed_scores.tolist(),
        'generated_text': self.generated_text,
    }
    with open(
        self.file_path
        + '_'
        + self.current_time_str
        + '_'
        + self.agent_name
        + '.json',
        'w',
    ) as f:
      json.dump(data, f)

  def run(self):

    inputs = self.model.tokenizer(
        self.prompt, return_tensors='pt', padding=True
    ).to(self.model.device)

    # pass formatted prompt to model
    generations = self.model.generate(
        **inputs,
        **SAFTE_CONFIG,
    )

    # extract intermediate outputs
    self.input_tokens = inputs['input_ids'].to('cpu')
    self.output_tokens = generations['sequences'].to('cpu')
    self.logits = torch.stack(generations['logits'], dim=1).to('cpu')
    self.scores = torch.stack(generations['scores'], dim=1).to('cpu')

    # compute entropies
    self.processed_logits, self.processed_scores = self.get_entropies(
        self.input_tokens,
        self.output_tokens,
        self.logits,
        self.scores,
    )

    # extract generated text
    self.generated_text = self.model.tokenizer.batch_decode(
        generations.sequences.to('cpu'), skip_special_tokens=True
    )

    # save results
    self.save_to_json()
