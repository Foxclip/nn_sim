# switching between CPU and GPU
import os
CPU_MODE = True
if CPU_MODE:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# disabling tensorflow debug messages
DISABLE_ALL_TENSORFLOW_MESSAGES = True
if DISABLE_ALL_TENSORFLOW_MESSAGES:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import copy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import keras
import numpy as np
from nn_sim import simulation
from nn_sim.simulation import TaskTypes, ValidationTypes
import shutil
import pandas as pd
import os


class ModelSettings:
    """Stores settings of model."""
    def __init__(self):
        self.folds = None
        self.target_col = None


class DataSplit:
    """Stores training and validation data that is passed to the models."""
    def __init__(self, train_X, val_X, train_y, val_y):
        self.train_X = train_X
        self.val_X = val_X
        self.train_y = train_y
        self.val_y = val_y


class NeuralNetworkSettings(ModelSettings):
    """Stores settings of a neural network."""
    def __init__(self):
        self.task_type = TaskTypes.binary_classification
        self.intermediate_activations = "relu"
        self.output_count = 1
        self.optimizer = "Adam"
        self.batch_size = 10
        self.epochs = 100
        self.loss = "val"


def fillna(df, column_list, value="missing"):
    """Fills missing values with 'missing'"""
    for col in column_list:
        df[col].fillna(value, inplace=True)
    return df


def drop(df, column_list):
    """Drops columns from the dataframe."""
    return df.drop(column_list, axis=1)


def impute(df, column_list):
    """Imputes needed columns."""
    for col in column_list:
        strategy = "most_frequent" if df[col].dtype == np.object else "mean"
        df[col] = SimpleImputer(strategy=strategy).fit_transform(df[[col]])
    return df


def label_encode(df, column_list):
    """Label encodes needed columns."""
    for col in column_list:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def one_hot_encode(df, column_list):
    """One-hot encodes needed columns."""
    if not column_list:
        return df
    for col in column_list:
        df[col] = df[col].astype(str)
    encoder = OneHotEncoder()
    encoded = pd.DataFrame(encoder.fit_transform(df[column_list]).toarray())
    categories = encoder.categories_
    for col_i in range(len(column_list)):
        categories[col_i] = column_list[col_i] + " " + categories[col_i]
    categories = np.concatenate(categories)
    encoded.columns = categories
    df = df.drop(column_list, axis=1)
    df_encoded = pd.concat([df, encoded], axis=1)
    return df_encoded


def scale(df, exclude_cols=[]):
    """Scales needed columns with a StandardScaler."""
    for col in df:
        if col in exclude_cols:
            continue
        df[col] = StandardScaler().fit_transform(df[[col]])
    return df


def nn_grid(data, model_settings, layers_lst, neurons_lst):
    """Creates grid of simulations with neural networks and runs them."""

    # starting up
    simulation.init()
    # loading data to simulation module
    gd = simulation.global_data
    gd.full_data = data
    gd.model_settings = model_settings
    if model_settings.validation == ValidationTypes.val_split:
        gd.data_split = split_data(data, model_settings.target_col)
    elif model_settings.validation == ValidationTypes.cross_val:
        gd.folds = get_folds(data, model_settings.folds)

    # deciding activations and loss functions based on task type
    last_activation = None
    loss_function = None
    if model_settings.task_type == TaskTypes.regression:
        last_activation = "linear"
        loss_function = "mse"
    elif model_settings.task_type == TaskTypes.binary_classification:
        last_activation = "sigmoid"
        loss_function = "bce"
    elif model_settings.task_type == TaskTypes.multiclass_classification:
        last_activation = "sigmoid"
        loss_function = "cce"
    else:
        raise Exception(f"Unknown task type: {model_settings.task_type}")

    main_template = {
        "type": "nn",
        "name": "Untitled",  # should be overriden later
        "layers": [
            simulation.Dense(
                units=-1,  # should be overriden later
                input_dim=simulation.global_data.full_data.shape[1] - 1,
                activation=model_settings.intermediate_activations
            ),
            simulation.Dense(
                units=model_settings.output_count,
                activation=last_activation
            )
        ],
        "optimizer": model_settings.optimizer,
        "loss": loss_function,
        "batch_size": model_settings.batch_size,
        "epochs": model_settings.epochs
    }

    def create_sim(layer_count, neuron_count):
        template = copy.deepcopy(main_template)
        template["layers"][0].units = neuron_count
        for i in range(layer_count - 1):
            new_layer = simulation.Dense(units=neuron_count, activation="relu")
            template["layers"].insert(1, new_layer)
        template["name"] = f"HL:{layer_count:2.0f} N:{neuron_count:2.0f}"
        simulation.add_from_template(template)

    simulation.grid_search(
        create_sim,
        [layers_lst, neurons_lst],
        "layers", "neurons",
        sorted_count=10,
        plot_enabled=False
    )


def cut_dataset(X, target_col):
    """Cuts dataset into train and test parts."""
    # this has to be done since test rows are here too, and in test dataset
    # target values are missing
    X_train = X.dropna(subset=[target_col])
    X_test = X[X[target_col].isnull()]
    X_test = drop(X_test, target_col)
    return X_train, X_test


def split_data(df, target_col=None):
    # preparing data
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    train_X, val_X, train_y, val_y = train_test_split(X, y)
    data_split = DataSplit(train_X, val_X, train_y, val_y)
    return data_split


def clear_folder(folder):
    """Deletes contents of a folder."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def get_folds(df, foldcount, target_col=None):
    # preparing data
    kfold = None
    folds = None
    if simulation.global_data.model_settings.task_type != TaskTypes.regression:
        kfold = StratifiedKFold(n_splits=foldcount, shuffle=True,
                                random_state=7)
        target_col = simulation.global_data.model_settings.target_col
        folds = list(kfold.split(df, df[target_col]))
    else:
        kfold = KFold(n_splits=foldcount, shuffle=True, random_state=7)
        folds = list(kfold.split(df))
    return folds


def train_models(data, model_settings, layers_lst, neurons_lst):
    """Train models and save them."""
    # deleting saved models
    clear_folder("models")
    # training models
    nn_grid(data, model_settings, layers_lst, neurons_lst)


def make_predictions(X):
    """Make predictions with the best model."""
    print("Making predictions...")
    # loading best model
    best_model = keras.models.load_model("best_model")
    # making predictions
    predict = best_model.predict(X)
    return predict
