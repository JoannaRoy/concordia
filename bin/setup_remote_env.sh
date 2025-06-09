rm -rf .venv

pt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.11 python3.11-venv python3.11-dev


python3.11 -m venv .venv
source .venv/bin/activate

pip install pip-tools
pip-compile --allow-unsafe --extra=dev --generate-hashes --strip-extras examples/requirements.in setup.py --output-file=requirements.txt

pip install -r requirements.txt
