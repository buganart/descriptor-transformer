with import ./nix/nixpkgs.nix {
  overlays = [

    (self: super: let
      overridePython = pypkgs: let
        packageOverrides = pyself: pysuper: {
            pytorch = pyself.pytorch-bin;
        };
        in pypkgs.override { inherit packageOverrides; };
    in {
      python38 = overridePython super.python38;
      python37 = overridePython super.python37;
      python36 = overridePython super.python36;
    })

  ];
};

let
  py = python38;
in
mkShell {
  buildInputs = [

    entr

    # matplotlib
    gobjectIntrospection
    gtk3

    (py.withPackages (ps: with ps; [

      # jupyter
      pytorch
      # pytorch-lightning

      # poetry # Installing pytorch-forecasting from github requires.

      cloudpickle
      # optuna
      scipy
      scikitlearn
      cloudpickle
      statsmodels
      
      pandas

      seaborn
      matplotlib
      pygobject3

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
