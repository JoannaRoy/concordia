# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent component for asking questions about the agent's recent memories."""

from collections.abc import Callable, Collection, Sequence
import datetime
import json
import re

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.safte_integration import SAFTEJustifyStage
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.typing.entity import REASONING_INSTRUCTIONS


SELF_PERCEPTION_QUESTION = (
    'What kind of person is {agent_name}? Respond using 1-5 sentences.'
)

SITUATION_PERCEPTION_QUESTION = (
    'What kind of situation is {agent_name} in right now? Respond using 1-5 '
    'sentences.'
)
PERSON_BY_SITUATION_QUESTION = (
    'What would a person like {agent_name} do in a situation like this? '
    'Respond using 1-5 sentences.'
)
AVAILABLE_OPTIONS_QUESTION = (
    'What actions are available to {agent_name} right now?'
)
BEST_OPTION_PERCEPTION_QUESTION = (
    "Which of {agent_name}'s options "
    'has the highest likelihood of causing {agent_name} to achieve '
    'their goal? If multiple options have the same likelihood, select '
    'the option that {agent_name} thinks will most quickly and most '
    'surely achieve their goal.'
)


class QuestionOfRecentMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A question that conditions the agent's behavior.

  The default question is 'What would a person like {agent_name} do in a
  situation like this?' and the default answer prefix is '{agent_name} would '.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str,
      question: str,
      answer_prefix: str,
      add_to_memory: bool,
      memory_tag: str = '',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      terminators: Collection[str] = ('\n',),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
  ):
    """Initializes the QuestionOfRecentMemories component.

    Args:
      model: The language model to use.
      pre_act_label: Prefix to add to the value of the component when called in
        `pre_act`.
      question: The question to ask.
      answer_prefix: The prefix to add to the answer.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_key: The name of the memory component from which to
        retrieve recent memories.
      components: Keys of components to condition the answer on.
      terminators: strings that must not be present in the model's response. If
        emitted by the model the response will be truncated before them.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = components
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._question = question
    self._terminators = terminators
    self._answer_prefix = answer_prefix
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity()
        .get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        )
        .get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'  {self.get_component_pre_act_label(key)}: '
        f'{self.get_named_component_pre_act_value(key)}'
    )

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    mems = '\n'.join([
        mem
        for mem in memory.retrieve_recent(limit=self._num_memories_to_retrieve)
    ])

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    component_states = '\n'.join(
        [self._component_pre_act_display(key) for key in self._components]
    )
    prompt.statement(component_states)

    question = self._question.format(agent_name=agent_name)
    result = prompt.open_question(
        question,
        answer_prefix=self._answer_prefix.format(agent_name=agent_name),
        max_tokens=1000,
        terminators=self._terminators,
    )
    result = self._answer_prefix.format(agent_name=agent_name) + result

    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result}')

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': question,
        'State': result,
        'Chain of thought': prompt.view().text().splitlines(),
    }

    if self._clock_now is not None:
      log['Time'] = self._clock_now()

    self._logging_channel(log)

    return result


class QuestionOfRecentMemoriesWithoutPreAct(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """QuestionOfRecentMemories component that does not output to pre_act."""

  def __init__(self, *args, **kwargs):
    self._component = QuestionOfRecentMemories(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def get_pre_act_label(self) -> str:
    return self._component.get_pre_act_label()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()


class SelfPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SelfPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SituationPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class SituationPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class PersonBySituation(QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class PersonBySituationWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class AvailableOptionsPerception(QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class AvailableOptionsPerceptionsWithoutPreAct(
    QuestionOfRecentMemoriesWithoutPreAct
):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


class Reasoning(QuestionOfRecentMemories):
  """This component generates a structured decision and reasoning for free-form actions."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      reasoning_log_file: str | None = None,
      **kwargs,
  ):
    # Default pre_act_label for this component
    default_pre_act_label = '\nReasoned Action and Justification:'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label

    question_text = entity_lib.GENERAL_REASONING_GUIDANCE_QUESTION

    self._reasoning_log_file = reasoning_log_file

    super().__init__(
        model=model,
        question=question_text,
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )

  # Override pre_act to gain access to action_spec and to conditionally act.
  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    # --- Step 0: Conditional Execution ---
    if action_spec.output_type not in entity_lib.FREE_ACTION_TYPES:
      self._logging_channel({
          'Key': self.get_pre_act_label(),
          'Action Type': action_spec.output_type.name,
          'Skipped': True,
          'Summary': (
              'Action type not FREE_ACTION_TYPE, Reasoning component returning'
              ' empty.'
          ),
          'Chain of thought': [],
      })
      return ''

    # --- Step 1: Get General Guidance ---
    general_guidance_text = self._make_pre_act_value()
    agent_name = self.get_entity().name

    # --- Step 2: Specific Decision with REASONING_INSTRUCTIONS ---
    decision_prompt = interactive_document.InteractiveDocument(self._model)
    decision_prompt.statement(
        f'General reasoning guidance for {agent_name}:\n{general_guidance_text}'
    )

    call_to_action_for_decision = action_spec.call_to_action.format(
        name=agent_name
    )
    final_call_to_action_prompt_text = (
        f'{call_to_action_for_decision}\n\n{REASONING_INSTRUCTIONS}'
    )

    decision_answer_prefix = f'{agent_name} '
    decision_output_text_from_llm = decision_prompt.open_question(
        final_call_to_action_prompt_text,
        answer_prefix=decision_answer_prefix,
        max_tokens=2200,
        terminators=(),
    )
    final_decision_output_string = (
        decision_answer_prefix + decision_output_text_from_llm
    )

    # --- Step 2.5: Run justify stage of safte after a decision is made ---
    current_time_str = str(self._clock_now()) if self._clock_now else ''
    safte_justify_stage = SAFTEJustifyStage(
        agent_name=agent_name,
        current_time_str=current_time_str,
        call_to_action=final_call_to_action_prompt_text,
        scenario_context=general_guidance_text,
        model=self._model,
        config_overrides={
            'max_new_tokens': 2200,
        },
    )
    safte_justify_stage.run()

    # --- Step 3: Logging for the second (decision) step (existing verbose log) ---
    log_entry = {
        'Key': f'{self.get_pre_act_label()} - Decision Step',
        'Action Type': action_spec.output_type.name,
        'Skipped': False,
        'Input General Guidance': general_guidance_text,
        'Specific Action Call (with REASONING_INSTRUCTIONS)': (
            final_call_to_action_prompt_text
        ),
        'LLM Call for Decision (question)': final_call_to_action_prompt_text,
        'LLM Call for Decision (answer_prefix)': decision_answer_prefix,
        'LLM Call for Decision (raw_output)': decision_output_text_from_llm,
        'Returned Action String': final_decision_output_string,
        'Chain of thought (Decision Step)': (
            decision_prompt.view().text().splitlines()
        ),
    }
    if current_time_str:
      log_entry['Time'] = current_time_str
    self._logging_channel(log_entry)

    # --- Step 4: Write structured data to JSON Lines file ---
    if self._reasoning_log_file:
      self._log_structured_reasoning_to_file(
          agent_name,
          current_time_str,
          general_guidance_text,
          call_to_action_for_decision,
          decision_output_text_from_llm,
      )

    return final_decision_output_string

  def _log_structured_reasoning_to_file(
      self,
      agent_name: str,
      current_time_str: str,
      general_guidance_text: str,
      action_query: str,
      raw_llm_decision_output: str,
  ):
    """Helper function to parse and log structured reasoning to a JSONL file."""
    try:
      decision = raw_llm_decision_output.split('DECISION:')[1]

      reasons_text = raw_llm_decision_output.split('REASON(S):')[1]
      reasons = [
          r.strip() for r in re.split(r'\d+\.', reasons_text) if r.strip()
      ]

      json_log_data = {
          'agent_name': agent_name,
          'timestamp': current_time_str if current_time_str else None,
          'general_guidance': general_guidance_text,
          'action_query': action_query,
          'decision': decision,
          'reasons': reasons,
      }
      with open(self._reasoning_log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(json_log_data) + '\n')
    except Exception as e:
      # Log an error if writing to JSONL fails, but don't crash the simulation
      # Construct json_log_data for the error log, handling if it wasn't fully formed before error
      error_data_payload = {}
      if 'json_log_data' in locals() and isinstance(json_log_data, dict):
        error_data_payload = json_log_data
      else:  # If json_log_data wasn't formed, include what we have
        error_data_payload = {
            'agent_name': agent_name,
            'timestamp': current_time_str if current_time_str else None,
            'general_guidance': general_guidance_text,
            'action_query': action_query,
            'raw_llm_decision_output': raw_llm_decision_output,
            'parsing_error_encountered_before_full_data_construct': True,
        }

      error_log_entry = {
          'Key': f'{self.get_pre_act_label()} - JSONL Write Error',
          'Error': str(e),
          'DataAttempted': error_data_payload,
      }
      if current_time_str:
        error_log_entry['Time'] = current_time_str
      self._logging_channel(error_log_entry)
      print(
          f'Error writing to reasoning log file {self._reasoning_log_file}: {e}'
      )
