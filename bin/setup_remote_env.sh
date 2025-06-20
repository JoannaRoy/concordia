#!/bin/bash

set -e

RUN_SCRIPT=false
HELP=false

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup remote environment with Python 3.11 and dependencies.

OPTIONS:
    --run-script    Run the simple example script after setup
    -h, --help      Show this help message and exit

EXAMPLES:
    $0                  # Basic setup only
    $0 --run-script     # Setup and run example script
    $0 --help           # Show this help

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --run-script)
            RUN_SCRIPT=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if [[ "$HELP" == true ]]; then
    usage
    exit 0
fi

rm -rf .venv

ssh-keygen -t ed25519 -C "runpod@runpod.io" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub | pbcopy
echo "Please paste the SSH key into GitHub and press Enter to continue..."
read

apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate

pip install pip-tools
pip-compile --allow-unsafe --extra=dev --generate-hashes --strip-extras examples/requirements.in setup.py --output-file=requirements.txt

pip install -r requirements.txt

if [[ "$RUN_SCRIPT" == true ]]; then
    python3 caif_project_evaluating_agent_safte/resources/python/simple_example.py
fi
