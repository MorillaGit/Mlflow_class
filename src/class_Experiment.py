from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, callbacks

# from src import data_utils, models, config
import os
import tensorflow as tf
import mlflow

import mlflow.tensorflow
import mlflow.keras
from .models import *


class Experiment:
    experiment_name: str
    model_name: str
    input_shape: tuple
    model: tf.keras.Model
    hyperparameters: dict
    metrics: dict
    ruta_carpeta_train: str
    ruta_carpeta_test: str
    batch_size: int
    epochs: int
    data_augmentation: bool
    kwargs: dict

    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        model: tf.keras.Model,
        hyperparameters: dict,
        metrics: dict,
        ruta_carpeta_train: str,
        ruta_carpeta_test: str,
        batch_size: int,
        epochs: int,
        input_shape: tuple = (224, 224, 3),
        data_augmentation: bool = False,
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.model = model
        self.hyperparameters = hyperparameters
        self.metrics = metrics
        self.ruta_carpeta_train = ruta_carpeta_train
        self.ruta_carpeta_test = ruta_carpeta_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_shape = input_shape
        self.data_augmentation = data_augmentation
        self.kwargs = kwargs

    def set_experiment_name(self, experiment_name: str):
        # Check if the experiment exists
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            # If it exists, set the experiment
            mlflow.set_experiment(experiment.name)
        else:
            # If it does not exist, create the experiment and set it
            mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)

    def run(self):
        # Configurar Data Augmentation
        train_datagen = (
            ImageDataGenerator(
                rotation_range=self.kwargs.get("rotation_range", 0),
                width_shift_range=self.kwargs.get("width_shift_range", 0),
                height_shift_range=self.kwargs.get("height_shift_range", 0),
                shear_range=self.kwargs.get("shear_range", 0),
                zoom_range=self.kwargs.get("zoom_range", 0),
                horizontal_flip=self.kwargs.get("horizontal_flip", False),
                fill_mode=self.kwargs.get("fill_mode", "nearest"),
            )
            if self.data_augmentation
            else ImageDataGenerator()
        )
        test_datagen = ImageDataGenerator()
        test_ds = test_datagen.flow_from_directory(
            self.ruta_carpeta_test,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        # Cargar datos de entrenamiento
        train_generator = train_datagen.flow_from_directory(
            self.ruta_carpeta_train,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode="categorical",
        )

        # Entrenar modelo
        mlflow.tensorflow.autolog(log_models=False)
        history = self.model.fit(
            train_generator,
            validation_data=test_ds,
            epochs=self.epochs,
            callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )
        return history

    def log_params(self):
        mlflow.log_param("model_name", self.model_name)
        for key, value in self.hyperparameters.items():
            mlflow.log_param(key, value)
        if self.data_augmentation:
            mlflow.log_params(self.kwargs)

    def log_metrics(self, history):
        for key, value in history.history.items():
            mlflow.log_metric(key, value[-1])

        # # Almacenar la métrica val_acc
        # val_acc = history.history["val_accuracy"][-1]
        # mlflow.log_metric("val_acc", val_acc)


def registrar_experiment(experiment: Experiment, experiment_name: str = None):
    with mlflow.start_run(nested=True):
        experiment.set_experiment_name(experiment_name)
        history = experiment.run()
        experiment.log_params()
        experiment.log_metrics(history)
