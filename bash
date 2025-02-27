set -Ux PYENV_ROOT $HOME/.pyenv
set -Ux PATH $PYENV_ROOT/bin $PATH
status --is-interactive; and source (pyenv init --path | psub)
