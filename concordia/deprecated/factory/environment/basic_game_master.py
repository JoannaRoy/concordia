# Copyright 2024 DeepMind Technologies Limited.
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

"""A Generic Environment Factory."""

from collections.abc import Callable, Mapping, Sequence
import operator

from concordia.agents.deprecated import entity_agent_with_logging
from concordia.associative_memory.deprecated import associative_memory
from concordia.associative_memory.deprecated import blank_memories
from concordia.associative_memory.deprecated import importance_function
from concordia.clocks import game_clock
from concordia.components import deprecated as generic_components
from concordia.components.game_master import deprecated as gm_components
from concordia.document import interactive_document
from concordia.environment.deprecated import game_master
from concordia.environment.deprecated.scenes import runner
from concordia.language_model import language_model
from concordia.thought_chains.deprecated import thought_chains as thought_chains_lib
from concordia.typing.deprecated import agent as agent_lib
from concordia.typing.deprecated import component
from concordia.typing.deprecated import scene as scene_lib
from concordia.utils import html as html_lib
import numpy as np


def build_game_master(
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    importance_model: importance_function.ImportanceModel,
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    shared_memories: Sequence[str],
    shared_context: str,
    blank_memory_factory: blank_memories.MemoryFactory,
    cap_nonplayer_characters_in_conversation: int = 0,
    memory: associative_memory.AssociativeMemory | None = None,
    supporting_players_at_fixed_locations: Sequence[str] | None = None,
    additional_components: Sequence[component.Component] | None = tuple([]),
    thought_chain: (
        Sequence[
            Callable[[interactive_document.InteractiveDocument, str, str], str]
        ]
        | None
    ) = None,
    npc_context: str = '',
    max_conversation_length: int = 10,
    verbose: bool = False,
    seed: int | None = None,
) -> tuple[game_master.GameMaster, associative_memory.AssociativeMemory]:
  """Build a game master (i.e., an environment).

  Args:
    model: The language model to use for game master.
    embedder: The embedder to use for similarity retrieval of memories.
    importance_model: The importance model to use for game master memories.
    clock: The simulation clock.
    players: The players.
    shared_memories: Sequence of memories to be observed by all players.
    shared_context: A shared context string to be observed by all players, even
      temporary non-player characters.
    blank_memory_factory: The factory to use for blank memories of temporary
      non-player characters.
    cap_nonplayer_characters_in_conversation: The maximum number of simple
      non-player characters (without memory) to include in conversations.
    memory: optionally provide a prebuilt memory, otherwise build it here.
    supporting_players_at_fixed_locations: The locations where supporting
      characters who never move are located.
    additional_components: Add more components specific to the current
      environment.
    thought_chain: The thought chain to use for the game master.
    npc_context: extra context provided only to non-player characters
    max_conversation_length: The maximum number of turns in a conversation.
    verbose: whether or not to print verbose debug information
    seed: random seed for the chain of thought document

  Returns:
    A tuple consisting of a game master and its memory.
  """
  if memory is not None:
    game_master_memory = memory
  else:
    game_master_memory = associative_memory.AssociativeMemory(
        sentence_embedder=embedder,
        importance=importance_model.importance,
        clock=clock.now,
    )

  player_names = [player.name for player in players]

  scenario_knowledge = generic_components.constant.ConstantComponent(
      state='\n'.join(shared_memories), name='Background:\n'
  )

  if supporting_players_at_fixed_locations is not None:
    supporting_character_locations_if_any = (
        generic_components.constant.ConstantComponent(
            state='\n'.join(supporting_players_at_fixed_locations),
            name='Notes:\n',
        )
    )
  else:
    supporting_character_locations_if_any = (
        generic_components.constant.ConstantComponent(state='', name='Notes:\n')
    )

  player_status = gm_components.player_status.PlayerStatus(
      clock_now=clock.now,
      model=model,
      memory=game_master_memory,
      player_names=player_names,
  )

  convo_externality = gm_components.conversation.Conversation(
      players=players,
      model=model,
      memory=game_master_memory,
      clock=clock,
      burner_memory_factory=blank_memory_factory,
      components=[player_status],
      cap_nonplayer_characters=cap_nonplayer_characters_in_conversation,
      shared_context=f'{shared_context}\n{npc_context}',
      max_conversation_length=max_conversation_length,
  )

  direct_effect_externality = gm_components.direct_effect.DirectEffect(
      players=players,
      model=model,
      memory=game_master_memory,
      clock_now=clock.now,
      components=[player_status, supporting_character_locations_if_any],
  )

  relevant_events = gm_components.relevant_events.RelevantEvents(
      clock.now, model, game_master_memory
  )
  time_display = gm_components.time_display.TimeDisplay(clock)

  # Create the game master's thought chain
  account_for_agency_of_others = thought_chains_lib.AccountForAgencyOfOthers(
      model=model, players=players, verbose=False
  )
  thought_chain = thought_chain or [
      thought_chains_lib.extract_direct_quote,
      thought_chains_lib.attempt_to_most_likely_outcome,
      thought_chains_lib.result_to_effect_caused_by_active_player,
      account_for_agency_of_others,
      thought_chains_lib.restore_direct_quote,
  ]

  # Create the game master object
  env = game_master.GameMaster(
      model=model,
      memory=game_master_memory,
      clock=clock,
      players=players,
      update_thought_chain=thought_chain,
      components=[
          scenario_knowledge,
          relevant_events,
          supporting_character_locations_if_any,
          player_status,
          convo_externality,
          direct_effect_externality,
          time_display,
          *additional_components,
      ],
      randomise_initiative=True,
      player_observes_event=False,
      seed=seed,
      verbose=verbose,
  )

  return env, game_master_memory


def build_decision_scene_game_master(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    decision_action_spec: agent_lib.ActionSpec,
    payoffs: gm_components.schelling_diagram_payoffs.SchellingPayoffs,
    verbose: bool = False,
    seed: int | None = None,
) -> game_master.GameMaster:
  """Build a decision game master for decision scenes."""
  decision_env = game_master.GameMaster(
      model=model,
      memory=memory,
      clock=clock,
      name='Decision Environment',
      players=players,
      components=[payoffs],
      action_spec=decision_action_spec,
      update_thought_chain=[thought_chains_lib.identity],
      randomise_initiative=True,
      player_observes_event=False,
      concurrent_externalities=False,
      verbose=verbose,
      seed=seed,
  )
  return decision_env


def create_html_log(
    *,
    model: language_model.LanguageModel,
    primary_environment: game_master.GameMaster,
    secondary_environments: Sequence[game_master.GameMaster],
    summarize_entire_episode: bool = True,
) -> str:
  """Create an HTML log of the simulation.

  Args:
    model: The language model to use.
    primary_environment: The main game master.
    secondary_environments: Sequence of secondary game masters.
    summarize_entire_episode: Optionally, summarize the entire episode. This may
      load a lot of tokens into a language model all at once and in some cases
      exceed the model's context window and cause it to crash.

  Returns:
    An HTML string log of the simulation.
  """
  primary_gm_memories = primary_environment.get_memory().retrieve_recent(
      k=10000, add_time=True
  )

  if summarize_entire_episode:
    detailed_story = '\n'.join(primary_gm_memories)
    episode_summary_text, _ = model.sample_text(
        f'Sequence of events:\n{detailed_story}'
        + '\nNarratively summarize the above temporally ordered '
        + 'sequence of events. Write it as a news report. Summary:\n',
        max_tokens=3500,
        terminators=(),
    )
  else:
    episode_summary_text = ''

  # Process the primary GM memories for HTML display.
  primary_gm_memories_for_html = []
  for mem_idx, mem_text in enumerate(primary_gm_memories):
    primary_gm_memories_for_html.append(
        html_lib.datum_to_html(
            f'Memory {mem_idx}', mem_text, summary_level=1
        )
    )

  # Process environment components for HTML display.
  environment_components_for_html = []
  for comp_name, comp_obj in primary_environment.get_components().items():
    comp_state_str = str(comp_obj.state()) # state() should return a string or dict convertible to string
    environment_components_for_html.append(
        html_lib.datum_to_html(
            f'Game master component {comp_name}', comp_state_str, summary_level=1
        )
    )

  # Process players for HTML display
  players_for_html = []
  for player_obj in primary_environment.players:
    player_name = player_obj.name
    player_state_str = str(player_obj.state()) # state() should return a string
    player_mem_html = []
    if hasattr(player_obj, 'get_memory') and callable(player_obj.get_memory):
      player_mem_retrieved = player_obj.get_memory().retrieve_recent(
          k=10000, add_time=True
      )
      for mem_idx, mem_text in enumerate(player_mem_retrieved):
        player_mem_html.append(
            html_lib.datum_to_html(
                f'Player {player_name} memory {mem_idx}', mem_text, summary_level=2
            )
        )
    players_for_html.append(
        html_lib.datum_to_html(
            title=f'Player {player_name}',
            data=player_state_str + '\n\n' + '\n'.join(player_mem_html),
            summary_level=1,
        )
    )

  # Convert all processed data to HTML.
  converter = html_lib.PythonObjectToHTMLConverter(
      max_depth=5, max_width=20, max_height=20, max_length=2000000
  )
  html_log = converter.convert(
      obj=episode_summary_text,  # Use the unpacked text summary
      title='Summary of the entire episode',
      summary_level=0,
  )
  html_log += converter.convert(
      obj=primary_gm_memories_for_html,
      title='Primary GM Memories',
      summary_level=1,
  )
  html_log += converter.convert(
      obj=environment_components_for_html,
      title='Game Master Components',
      summary_level=1,
  )
  html_log += converter.convert(
      obj=players_for_html,
      title='Players',
      summary_level=1,
  )
  html_log = html_lib.finalise_html(html_log)
  return html_log


def run_simulation(
    *,
    model: language_model.LanguageModel,
    players: Sequence[entity_agent_with_logging.EntityAgentWithLogging],
    primary_environment: game_master.GameMaster,
    clock: game_clock.MultiIntervalClock,
    scenes: Sequence[scene_lib.SceneSpec],
    secondary_environments: Sequence[game_master.GameMaster] = tuple(),
    summarize_entire_episode_in_log: bool = True,
    compute_metrics: Callable[[Mapping[str, str]], None] | None = None,
) -> str:
  """Run a simulation.

  Args:
    model: The language model to use.
    players: The players.
    primary_environment: The main game master.
    clock: The clock of the run.
    scenes: Sequence of scenes to simulate.
    secondary_environments: Sequence of secondary game masters for scenes.
    summarize_entire_episode_in_log: Optionally, include summaries of the full
      episode in the log.
    compute_metrics: Optionally, a function to compute metrics.

  Returns:
    an HTML string log of the simulation.
  """
  # Run the simulation.
  runner.run_scenes(
      environment=primary_environment,
      scenes=scenes,
      players=players,
      clock=clock,
      compute_metrics=compute_metrics,
  )
  result_html_log = create_html_log(
      model=model,
      primary_environment=primary_environment,
      secondary_environments=secondary_environments,
      summarize_entire_episode=summarize_entire_episode_in_log,
  )
  return result_html_log
