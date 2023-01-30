# Python ML project template 

This is an opinionated project template for a Python based machine learning projects.  
This template is to be used as the default project start for any Python ML project (unless arguments support otherwise).  
While the project is heavily opinionated, opinions are welcomed to be discussed: feel free to open an issue or comment.

## Installing the package

1. Clone the repo

    ```bash
    git clone git@github.com:datarootsio/ml_template_hydra.git
    cd ml_template_hydra
    ```

2. Install dependencies using [pip](https://pip.pypa.io/en/stable/installing/). The following command
will install the dependencies from `setup.py`. In the backend it will run `pip install -e ".[test, serve]"`. Note that installing dependencies with `-e` 
editable mode is needed to properly run unit tests. `[test, serve]` is optional. `test` refers to
unit test dependencies and `serve` refers to deployment dependencies.

    ```bash
    make install
    ```

## Running the project

Preferably, you can use make commands (from `Makefile`) or directly run scripts from `scripts`.  
Refer to section below for the descriptions of make commands. Before running it, consider creating  
a virtual environment.  

**Makefile and test example**

Try out the `make` commands on the example `creditcard.csv` dataset model (see `make help`).

```
clean                          clean artifacts
coverage                       create coverage report
generate-dataset               run ETL pipeline
help                           show help on available commands
lint                           flake8 linting and black code style
run-pipeline                   clean artifacts -> generate dataset -> train -> serve
serve                          serve trained model with a REST API using dploy-kickstart
test-docker                    run unit tests in docker environment
test                           run unit tests in the current virtual environment
train                          train the model, you can pass arguments as follows: make ARGS="--foo 10 --bar 20" train
```

Note the dependency: `generate-dataset` > `train` > `serve`.

## Docker

Currently, you can find the following docker files:  
1. `jupyter.Dockerfile` builds an image for running notebooks.  
2. `test.Dockerfile` builds an image to run all tests in (`make test-docker`).
3. `serve.Dockerfile` build an image to serve the trained model via a REST api.
To ease the serving it uses open source `dploy-kickstart` module. To find more info
about `dploy-kickstart` click [here](https://github.com/dploy-ai/dploy-kickstart/).

Finally, you can start all services using `docker-compose`:  
for example `docker-compose up jupyter` or `docker-compose up serve`.  

Do you need a notebook for development? Just run `docker-compose up jupyter`. It will launch a Jupyter Notebook 
with access to your local development files.

## Project Structure Overview 
The project structure tree is shown below. This structure is designed
in a way to easily develop ML projects. Feedback / PRs are always welcome
about the structure.

```
.
├── .github             # Github actions CI pipeline
|
├── data                
│   ├── predictions     # predictions data, calculated using the model
│   ├── raw             # immutable original data
│   ├── staging         # data obtained after preprocessing, i.e. cleaning, merging, filtering etc.
│   ├── checkpoints     # Store all the serialized models here
│   └── transformed     # data ready for modeling (dataset containing features and label)
|   
├── configs              # Hydra Configs
|
├── docker              # Store all dockerfiles
|
├── notebooks           # Store prototype or exploration related .ipynb notebooks
|
├── reports             # Store textual or visualisation content, i.e. pdf, latex, .doc, .txt 
|
├── src             # Storing all the scripts here
|
└── tests               # Unit tests
```

## Best practices for development

- Make sure that `docker-compose up test` runs properly.  
- In need for a Notebook? Use the docker image: `docker-compose up jupyter`.
- Commit often, perfect later.
- Integrate `make test` with your CI pipeline.
- Capture `stdout` when deployed.
