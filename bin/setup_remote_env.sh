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
