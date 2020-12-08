with import ./nix/nixpkgs.nix {};

let
  py = python3;
in
mkShell {
  buildInputs = [

    entr

    (py.withPackages (ps: with ps; [

      jupyter
      pytorch-bin
      pandas


      tqdm
      scikitlearn

      librosa
      soundfile

      # 2020-08-07: wandb not yet available in nixpkgs
      pip

      black
      pre-commit

      pytest
      pytest-watch
      pytestcov

      # dev deps
      pudb  # debugger
      ipython
      pyls-isort
      pyls-mypy
      python-language-server
    ]))
   ];

  shellHook = ''
    export PIP_PREFIX="$(pwd)/.build/pip_packages"
    export PATH="$PIP_PREFIX/bin:$PATH"
    export PYTHONPATH="$PIP_PREFIX/${py.sitePackages}:$PYTHONPATH"
    unset SOURCE_DATE_EPOCH
  '';
}
