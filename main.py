import os
# switching between CPU and GPU
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import keras
import pandas as pd
import numpy as np
import simulation
import shutil
import os
import enum
import matplotlib.pyplot as plt


class DataSplit:
    """Stores training and validation data that is passed to the models."""
    def __init__(self, train_X, val_X, train_y, val_y, colcount):
        self.train_X = train_X
        self.val_X = val_X
        self.train_y = train_y
        self.val_y = val_y
        self.colcount = colcount


class ColNames:
    """Stores column names in one place."""
    def __init__(self):
        self.target_col = None
        self.test_id = None  # leftmost column in final output on test dataset
        self.fillna_cols = []
        self.impute_cols = []
        self.label_encode_cols = []
        self.onehot_encode_cols = []
        self.drop_cols = []


class TaskTypes(enum.Enum):
    """Types of tasks for neural networks."""
    regression = 0,
    binary_classification = 1,
    multiclass_classification = 2


class ModelSettings:
    """Stores settings of model."""
    def __init__(self):
        pass


class NeuralNetworkSettings(ModelSettings):
    """Stores settings of a neural network."""
    def __init__(self):
        self.task_type = TaskTypes.binary_classification
        self.intermediate_activations = "relu"
        self.output_count = 1
        self.optimizer = "Adam"
        self.batch_size = 10
        self.epochs = 100


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
        df[col] = SimpleImputer().fit_transform(df[[col]])
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


def find_best_randomstate(train_X, train_y, val_X, val_y):
    errors = []
    for rs_i in range(10000):
        if rs_i % 1000 == 0:
            print(rs_i)
        model = DecisionTreeClassifier(random_state=rs_i)
        model = model.fit(train_X, train_y)
        predict = model.predict(val_X)
        error = mean_absolute_error(val_y, predict)
        errors.append(error)
    min_index = np.argmin(errors)
    min_value = errors[min_index]
    return min_index, min_value


def save_result_trees(model):

    # column names
    fillna_cols = ["Cabin", "Embarked"]
    impute_cols = ["Age", "Fare"]
    label_encode_cols = ["Sex", "Cabin", "Embarked", "Ticket"]
    onehot_encode_cols = []
    drop_cols = ["Name", "PassengerId"]

    # loading test dataset
    df_test = pd.read_csv("test.csv")
    new_df = df_test[["PassengerId"]]

    # preparing test dataset
    X = df_test.copy()
    X = fillna(X, fillna_cols)
    X = impute(X, impute_cols)
    X = label_encode(X, label_encode_cols)
    X = one_hot_encode(X, onehot_encode_cols)
    X = drop(X, drop_cols)

    # making predictions on test dataset
    pred_test = model.predict(X)

    # making and saving new dataframe
    new_df = new_df.assign(Survived=pred_test)
    print(new_df)
    new_df.to_csv("result.csv", index=False)


def train_and_save_trees(data_split):

    # training model
    # model = DecisionTreeClassifier(max_depth=10, random_state=17)
    model = RandomForestClassifier(n_estimators=100, max_depth=11)
    model.fit(data_split.train_X, data_split.train_y)

    save_result_trees(model)

    # measuring error
    predict = model.predict(data_split.val_X)
    print(mean_absolute_error(data_split.val_y, predict))


def plot_avg_for_attr(attr_name, attr_lst):
    averages = []
    for attr in attr_lst:
        losses = [sim.loss
                  for sim
                  in simulation.simulations
                  if getattr(sim, attr_name) == attr]
        average = np.mean(losses)
        averages.append(average)
    plt.plot(attr_lst, averages)
    plt.show()


def simulate_trees(data_split):

    simulation.init()
    simulation.global_data.data_split = data_split

    main_template = {
        "count": 10,
        # "type": "decision_tree",
        "type": "random_forest",
        "leafcount": 10
    }
    leafcount_lst = list(range(1, 21))
    random_state_lst = list(range(1, 11))

    def create_sim(leafcount, random_state):
        template = copy.deepcopy(main_template)
        template["leafcount"] = leafcount
        template["random_state"] = random_state
        template["name"] = f"lc:{leafcount} rs:{random_state}"
        simulation.add_from_template(template)

    # simulation.sim_list(template_list)
    simulation.grid_search(
        create_sim,
        [leafcount_lst, random_state_lst],
        "leafcount", "random_state",
        sorted_count=10,
        plot_enabled=False
    )

    plot_avg_for_attr("leafcount", leafcount_lst)


def nn_grid(data_split, model_settings, layers_lst, neurons_lst):
    """Creates grid of simulations with neural networks and runs them."""

    # starting up
    simulation.init()
    # loading data to simulation module
    simulation.global_data.data_split = data_split

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
                input_dim=data_split.colcount,
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
        for i in range(layer_count):
            new_layer = simulation.Dense(units=neuron_count, activation="relu")
            template["layers"].insert(1, new_layer)
        template["name"] = f"HL:{layer_count} N:{neuron_count}"
        simulation.add_from_template(template)

    simulation.grid_search(
        create_sim,
        [layers_lst, neurons_lst],
        "layers", "neurons",
        sorted_count=10,
        plot_enabled=False
    )


def prepare(df, colnames, apply_scaling=False):
    """Runs various operations on the dataframe."""
    X = df.copy()
    X = fillna(X, colnames.fillna_cols)
    X = impute(X, colnames.impute_cols)
    X = label_encode(X, colnames.label_encode_cols)
    X = one_hot_encode(X, colnames.onehot_encode_cols)
    X = drop(X, colnames.drop_cols)
    if apply_scaling:
        X = scale(X, exclude_cols=[colnames.target_col])
    # this has to be done since test rows are here too, and in test dataset
    # target values are missing
    X_train = X.dropna(subset=[colnames.target_col])
    y_train = X_train[colnames.target_col]
    X_train = drop(X_train, colnames.target_col)
    X_test = X[X[colnames.target_col].isnull()]
    X_test = drop(X_test, colnames.target_col)
    return X_train, X_test, y_train


def prepare_for_trees(df):
    """Prepares the dataframe for trees."""
    global colnames
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = ["Sex", "Cabin", "Embarked", "Ticket"]
    colnames.onehot_encode_cols = []
    colnames.drop_cols = ["Name", "PassengerId"]
    return prepare(df, colnames)


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


def load_data(colnames, f):
    # loading CSV
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    # applying transformations (feature engineering)
    df = f(df)
    # preparing data
    X_train, X_test, y_train = prepare(df, colnames, apply_scaling=True)
    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train)
    data_split = DataSplit(train_X, val_X, train_y, val_y, train_X.shape[1])
    return data_split, X_test


def train_models(data_split, layers_lst, neurons_lst, epochs):
    """Train models and save them."""
    # deleting saved models
    clear_folder("models")
    # training models
    nn_grid(data_split, layers_lst, neurons_lst, epochs)


def make_predictions(X_test, colnames):
    """Make predictions with the best model."""
    print("Making predictions...")
    # loading test CSV
    df = pd.read_csv("test.csv")
    # loading best model
    best_model = keras.models.load_model("best_model")
    # making predictions
    predict = np.round(best_model.predict(X_test))
    df = df[[colnames.test_id]]
    df[colnames.target_col] = predict
    df[colnames.target_col] = df[colnames.target_col].astype(int)
    df.to_csv("output.csv", index=False)


if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 15)

    # defining dataset transormations (feature engineering)
    def transform_dataset(df):
        df["Family"] = df["SibSp"] + df["Parch"]
        return df

    # specifying what to do with dataset
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.test_id = "PassengerId"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = []
    colnames.onehot_encode_cols = ["Sex", "Embarked"]
    colnames.drop_cols = ["Name", "PassengerId", "Ticket", "Cabin", "SibSp",
                          "Parch"]

    # specifying settings of a model
    model_settings = NeuralNetworkSettings()
    model_settings.task_type = TaskTypes.binary_classification
    model_settings.intermediate_activations = "relu"
    model_settings.output_count = 1
    model_settings.optimizer = "Adam"
    model_settings.batch_size = 10
    model_settings.epochs = 5000

    # specifying lists of parameters
    layers_lst = [1, 2, 3]
    neurons_lst = [3, 4, 5]

    # loading and preparing data
    data_split, X_test = load_data(colnames, transform_dataset)

    # training models and saving file with predictions on test dataset
    train_models(data_split, model_settings, layers_lst, neurons_lst)

    # making predictions with the best model
    make_predictions(X_test, colnames)
