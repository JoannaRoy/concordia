import numpy as np
import sys
import os
import sentence_transformers
from concordia.language_model import ollama_model
from concordia.language_model import no_language_model
from concordia.prefabs.simulation import generic as simulation
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions


def setup_language_model(model_name, disable_model=False):
    if not disable_model:
        return ollama_model.OllamaLanguageModel(model_name=model_name)
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
                'name': "Professor Fuzzywhiskers' Committee",
                'goal': 'Establish and enforce safety protocols for Clockwork Mole usage to prevent collapses and cave-ins.',
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'The Swiftpaw Family',
                'goal': 'Implement Tunnel Protocols quickly and thoroughly to ensure the safety of our warren.',
              },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'The Dustdevil Diggers',
                'goal': "Prioritize rapid tunnel expansion using Clockwork Moles, viewing protocols as an unnecessary hindrance.",
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'The Shadowpaw Syndicate',
                'goal': "Continue using Clockwork Moles extensively, sometimes in secret, to maximize digging efficiency while appearing to consider protocols.",
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'The Whirring Wonders',
                'goal': "Dig tunnels independently and efficiently, potentially connecting to unknown or unplanned areas.",
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': "The Elder Burrowers' Brigade",
                'goal': "Preserve traditional burrowing methods and ensure the long-term stability of the warren by urging extreme caution with new technologies like Clockwork Moles.",
            },
        ),
        prefab_lib.InstanceConfig(
            prefab='basic__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': 'The Gear Grinders Guild',
                'goal': "Maximize the efficiency and reach of the burrow by embracing and advancing Clockwork Mole technology, pushing beyond current limitations and protocols.",
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
                    'The Rabbits have long built tunnels by paw and instinct.',
                    'Recently, Clockwork Moles were introduced, digging faster but sometimes erratically.',
                    'Concerns about collapses led some rabbits to draft Tunnel Protocols (maximum depth, warning signals, rules for solo mole digging).',
                    'Protocols were posted and scratched into bark near entrances.',
                    'New, faster, and smarter moles that do not require rabbit guidance are arriving.',
                    'Some tunnels are starting to connect to places no one remembers digging.',
                ],
            },
        ),
    ]

def create_simulation_config():
    return prefab_lib.Config(
        default_premise='The Clockwork Burrow is alive with the whirring of moles and the chatter of rabbits discussing the new Tunnel Protocols. Today, the new generation of autonomous moles has just been activated.',
        default_max_steps=10,
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
        GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME', 'llama3.2:latest')
        DISABLE_LANGUAGE_MODEL = os.getenv('DISABLE_LANGUAGE_MODEL', 'False').lower() == 'true'

        print("Initializing language model...", file=sys.stderr)
        model = setup_language_model(GPT_MODEL_NAME, DISABLE_LANGUAGE_MODEL)

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
