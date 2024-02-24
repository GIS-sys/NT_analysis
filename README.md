### something

sudo apt-get install gnumeric

ssconvert data/raw.xlsx data/raw.csv

prepare_data.py


# MLOps project

Predicting clothing types, training on FashionMNIST dataset

# Quick start

1) train: ```poetry run train```

2) infer: ```poetry run infer```

3) test: ```poetry run pytest```

4) run server: ```./run_server.sh```

5) test server by running ```python server/test.py```

6) triton info in triton/ folder (including readme with optimization info)

# Requirements

1) install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)

2) create new environment in conda and activate it

3) install poetry using conda

4) update poetry using ```pip install poetry -U```

5) install dependencies using poetry

# For development

Useful poetry comands:
- to build: poetry build
- to show project structure: poetry show --tree
- to install dependencies in current env: poetry lock; poetry install
- to run script: poetry run [script name from pyproject.toml]
- to run tests: poetry run pytest

Useful conda commands:
- to create environment with specific python version: conda create -n "myenv" python=3.3.0
- to permanently remove auto-base env: conda config --set auto\_activate\_base false
- to activate base env: conda activate
- to deactivate base env: conda deactivate

Useful pre-commit commands:
- to init pre-commit config in new repository: pre-commit install
- to run per-commit on all files: pre-commit run --all-files

Useful dvc commands:
- to init dvc: dvc init
- to add gdrive remote: dvc remote add -d storage gdrive://1Z3JfbS00SLrhHPVh7igikSy4Dbbug-z
- to add new files and push them: dvc add data/; dvc commit; dvc push
- to restore to current hash: dvc pull

Useful nginx commands:
- to add new nginx config: sudo vim /etc/nginx/sites-available/mlflow.conf
- to enable this new config: sudo ln -s /etc/nginx/sites-available/mlflow.conf /etc/nginx/sites-enabled/
- to add alias to your website (if it's not public): echo -e "\n127.0.0.1 gismlops.mlflow\n" | sudo tee -a /etc/hosts
- to check nginx configuration: sudo nginx -t
- to reload nginx: sudo systemctl restart nginx

Useful perf_analyzer commands:
- to launch perf_analyzer (from inside of triton/ folder): docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk
- to test concurrency: perf_analyzer -m onnx-clothing -u localhost:8500 --concurrency-range 1:5 --shape inputs:1,1,28,28 --shape predictions:1,10

To start working make sure you:
1) activated conda environment
2) if using web mlflow logging, make sure to run docker from tracker-service/ and export username and password:
```
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password
```

To serve model:
1) use ```server/build_server.sh``` to create .onnx model and Docker container
2) use ```server/start_server.sh``` to start existing server
*) or use ```./run_server.sh``` instead of previous two commands
3) test if everything ok by using ```poetry run server/test.py```

To serve model with triton: go to [triton/README.md](triton/README.md) for further info

## Troubleshooting

- if you get poetry error "Failed to create the collection. Prompt dismissed. .", then use
```export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring```
