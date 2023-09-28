# MLOps project

Egorov Gordei, MIPT, Б05-027

## Requirements

1) install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

2) create new environment in conda and activate it

3) install poetry using conda

4) update poetry using ```pip install poetry -U```

5) install dependencies using poetry

# dev

Useful poetry comands:
- to build: poetry build
- to show project structure: poetry show --tree
- to install dependencies in current env: poetry lock; poetry install
- to run script: poetry run [script name from pyproject.toml]

Useful conda commands:
- to permanently remove auto-base env: conda config --set auto\_activate\_base false
- to activate base env: conda activate
- to deactivate base env: conda deactivate

Useful pre-commit commands:
- to init pre-commit config in new repository: pre-commit install

To start working make sure you:
1) activated conda environment
