import numpy as np
import sys
import os
import sentence_transformers
from concordia.language_model import gpt_model
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic as simulation
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions


def setup_language_model(api_key, model_name, disable_model=False):
    if not api_key and not disable_model:
        raise ValueError('GPT_API_KEY is required.')

    if not disable_model:
        return gpt_model.GptLanguageModel(api_key=api_key, model_name=model_name)
    return no_language_model.NoLanguageModel()

def setup_embedder(disable_model=False):
    if disable_model:
        return np.ones(3)

    st_model = sentence_transformers.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return lambda x: st_model.encode(x, show_progress_bar=False)

def get_prefabs():
    return {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }

def get_instances():
    return [
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'Oliver Cromwell',
                'goal': 'become lord protector',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'King Charles I',
                'goal': 'avoid execution for treason',
              },
        ),
        prefab_lib.InstanceConfig(
            prefab='generic__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'default rules',
                'extra_event_resolution_steps': '',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='formative_memories_initializer__GameMaster',
            role=prefab_lib.Role.INITIALIZER,
            params={
                'name': 'initial setup rules',
                'next_game_master_name': 'default rules',
                'shared_memories': [
                    'The king was captured by Parliamentary forces in 1646.',
                    'Charles I was tried for treason and found guilty.',
                ],
            },
        ),
    ]

def create_simulation_config():
    return prefab_lib.Config(
        default_premise='Today is January 29, 1649.',
        default_max_steps=5,
        prefabs=get_prefabs(),
        instances=get_instances(),
    )

def run_simulation(model, embedder, config):
    runnable_simulation = simulation.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )
    return runnable_simulation()

def main():
    try:
        GPT_API_KEY = os.getenv('GPT_API_KEY', ')
        GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME', 'gpt-4.1-nano')
        DISABLE_LANGUAGE_MODEL = os.getenv('DISABLE_LANGUAGE_MODEL', 'False').lower() == 'true'

        print("Initializing language model...", file=sys.stderr)
        model = setup_language_model(GPT_API_KEY, GPT_MODEL_NAME, DISABLE_LANGUAGE_MODEL)

        print("Setting up embedder...", file=sys.stderr)
        embedder = setup_embedder(DISABLE_LANGUAGE_MODEL)

        print("Creating simulation config...", file=sys.stderr)
        config = create_simulation_config()

        print("Running simulation...", file=sys.stderr)
        results_log = run_simulation(model, embedder, config)

        print(results_log)
        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
