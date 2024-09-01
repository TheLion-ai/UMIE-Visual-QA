#compdef main.py

_mainpy_completion() {
  eval $(env _TYPER_COMPLETE_ARGS="${words[1,$CURRENT]}" _MAIN.PY_COMPLETE=complete_zsh main.py)
}

compdef _mainpy_completion main.py