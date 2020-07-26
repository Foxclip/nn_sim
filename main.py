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
        self.target_col = []
        self.fillna_cols = []
        self.impute_cols = []
        self.label_encode_cols = []
        self.onehot_encode_cols = []
        self.drop_cols = []


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


def scale(df):
    """Scales needed columns with a StandardScaler."""
    for col in df:
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


def nn_one(data_split, layer_count, neuron_count, epochs):
    """Creates one simulation with neural network and runs it."""

    simulation.init()
    simulation.global_data.data_split = data_split
    cols = data_split.colcount

    main_template = {
        "type": "nn",
        "name": "Untitled",
        "layers": [
            simulation.Dense(units=3, input_dim=cols, activation="relu"),
            simulation.Dense(units=1, activation="sigmoid")
        ],
        "optimizer": "Adam",
        "loss": "bce",
        "batch_size": 10,
        "epochs": epochs
    }
    templates = []
    template = copy.deepcopy(main_template)
    for i in range(layer_count - 1):
        new_layer = simulation.Dense(units=neuron_count, activation="relu")
        template["layers"].insert(1, new_layer)
    template["name"] = f"HL:{layer_count} N:{neuron_count} neurons"
    templates.append(template)

    simulation.sim_list(templates, plotting=[])


def nn_list(data_split, max_neurons, epochs):
    """Creates list of simulations with neural networks and runs them."""

    simulation.init()
    simulation.global_data.data_split = data_split
    cols = data_split.colcount

    main_template = {
        "type": "nn",
        "name": "Untitled",
        "layers": [
            simulation.Dense(units=3, input_dim=cols, activation="relu"),
            simulation.Dense(units=1, activation="sigmoid")
        ],
        "optimizer": "Adam",
        "loss": "bce",
        "batch_size": 10,
        "epochs": epochs
    }
    templates = []
    for ucount in range(1, max_neurons):
        template = copy.deepcopy(main_template)
        # new_layer = simulation.Dense(units=ucount, activation="relu")
        # template["layers"].insert(1, new_layer)
        template["name"] = f"{ucount} neurons"
        templates.append(template)

    simulation.sim_list(templates, plotting=[])


def nn_grid(data_split, max_layers, max_neurons, epochs):
    """Creates grid of simulations with neural networks and runs them."""

    simulation.init()
    simulation.global_data.data_split = data_split
    cols = data_split.colcount

    main_template = {
        "type": "nn",
        "name": "Untitled",
        "layers": [
            simulation.Dense(units=3, input_dim=cols, activation="relu"),
            simulation.Dense(units=1, activation="sigmoid")
        ],
        "optimizer": "Adam",
        "loss": "bce",
        "batch_size": 10,
        "epochs": epochs
    }

    layers_lst = list(range(0, max_layers + 1))
    neurons_lst = list(range(1, max_neurons + 1))

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


def prepare(df, colnames, apply_scaling=False, target=True):
    """Runs various operations on the dataframe."""
    X = df.copy()
    if target:
        X = drop(X, colnames.target_col)
    X = fillna(X, colnames.fillna_cols)
    X = impute(X, colnames.impute_cols)
    X = label_encode(X, colnames.label_encode_cols)
    X = one_hot_encode(X, colnames.onehot_encode_cols)
    X = drop(X, colnames.drop_cols)
    if apply_scaling:
        X = scale(X)
    if target:
        y = df[colnames.target_col]
        return X, y
    else:
        return X


def prepare_for_trees(df):
    """Prepares the dataframe for trees."""
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = ["Sex", "Cabin", "Embarked", "Ticket"]
    colnames.onehot_encode_cols = []
    colnames.drop_cols = ["Name", "PassengerId"]
    return prepare(df, colnames)


def prepare_for_nn(df, target=True):
    """Prepares the dataframe for neural networks."""
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = []
    colnames.onehot_encode_cols = ["Sex", "Embarked"]
    colnames.drop_cols = ["Name", "PassengerId", "Ticket", "Cabin"]
    return prepare(df, colnames, apply_scaling=True, target=target)


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


def train_models(layers, neurons, epochs):
    """Train models and save them."""
    # loading CSV
    df = pd.read_csv("train.csv")
    # preparing data
    X, y = prepare_for_nn(df)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    data_split = DataSplit(train_X, val_X, train_y, val_y, train_X.shape[1])
    # deleting saved models
    clear_folder("models")
    # training models
    nn_one(data_split, layers, neurons, epochs)
    # nn_grid(data_split, 5, 5, 1000)
    # saving column names
    df = pd.DataFrame(X.columns)
    df.columns = ["Column names"]
    df.to_csv("column_names.csv")


def make_predictions():
    """Make predictions with the best model."""
    print("Making predictions...")
    # loading test CSV
    df = pd.read_csv("test.csv")
    # preparing data
    X = prepare_for_nn(df, target=False)
    # filling missing columns with zeroes
    column_list = pd.read_csv("column_names.csv")["Column names"]
    missing_columns = [col for col in column_list if col not in X.columns]
    for col in missing_columns:
        X[col] = 0.0
    # loading best model
    best_model = keras.models.load_model("best_model")
    # making predictions
    predict = np.round(best_model.predict(X))
    df = df[["PassengerId"]]
    df["Survived"] = predict
    df["Survived"] = df["Survived"].astype(int)
    df.to_csv("output.csv", index=False)


if __name__ == "__main__":

    # Increasing number of columns so all of them are showed
    pd.set_option('display.max_columns', 15)

    train_models(1, 3, 10000)
    make_predictions()
