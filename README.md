# MLflow Classes for Credit Risk Prediction and Image Classification

This repository contains a set of classes for training machine learning models and logging experiments using [MLflow](https://mlflow.org/) library. The classes are designed to work with structured data for credit risk prediction and image classification problems.

## Usage

### Credit Risk Prediction

The repository contains a Jupyter notebook `main.ipynb` that demonstrates how to use the classes to train a credit risk prediction model using a structured dataset. The notebook shows how to:

- Load data and perform preprocessing
- Train and evaluate a machine learning model
- Log the parameters, metrics, and artifacts using MLflow

### Image Classification

The repository also contains a Jupyter notebook `image_classification.ipynb` that demonstrates how to use the classes to train an image classification model using a pre-trained EfficientNet model. The notebook shows how to:

- Load and preprocess images
- Fine-tune a pre-trained model
- Log the parameters, metrics, and artifacts using MLflow

## Classes

The repository contains the following classes:

- `CreditRiskExperiment`: A class for running a credit risk prediction experiment using a structured dataset.
- `ImageClassificationExperiment`: A class for running an image classification experiment using a pre-trained EfficientNet model.

Both classes have the following methods:

- `__init__`: Initializes the class and sets up the MLflow experiment.
- `run`: Runs the experiment and logs the parameters, metrics, and artifacts.
- `log_params`: Logs the parameters used in the experiment.
- `log_metrics`: Logs the metrics computed during the experiment.
- `log_artifacts`: Logs the artifacts generated during the experiment.

## Installation

To use the classes, first clone the repository:

```sh
git clone https://github.com/<username>/<repository-name>.git
