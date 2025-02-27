yay -S pyenv
yay -S python3.11

set -Ux PYENV_ROOT $HOME/.pyenv
set -Ux PATH $PYENV_ROOT/bin $PATH
status --is-interactive; and source (pyenv init --path | psub)


pip install --upgrade pip
pyenv install 3.11.6
pyenv local 3.11.6
pip install --no-cache-dir spacy

nvcc --version
