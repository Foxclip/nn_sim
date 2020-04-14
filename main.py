from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import simulation
import matplotlib.pyplot as plt


class DataSplit:
    def __init__(self, train_X, val_X, train_y, val_y):
        self.train_X = train_X
        self.val_X = val_X
        self.train_y = train_y
        self.val_y = val_y


class ColNames:
    def __init__(self):
        self.target_col = []
        self.fillna_cols = []
        self.impute_cols = []
        self.label_encode_cols = []
        self.onehot_encode_cols = []
        self.drop_cols = []


def fillna(df, column_list, value="missing"):
    for col in column_list:
        df[col].fillna(value, inplace=True)
    return df


def drop(df, column_list):
    return df.drop(column_list, axis=1)


def impute(df, column_list):
    for col in column_list:
        df[col] = SimpleImputer().fit_transform(df[[col]])
    return df


def label_encode(df, column_list):
    for col in column_list:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def one_hot_encode(df, column_list):
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
        template = main_template.copy()
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


def simulate_nn(data_split):

    simulation.init()
    simulation.global_data.data_split = data_split

    main_template = {
        "type": "nn",
        "name": "Utitled",
        "layers": [
            simulation.Dense(units=3, input_dim=11, activation="relu"),
            simulation.Dense(units=1, activation="linear")
        ],
        "optimizer": "Adam",
        "loss": "mse",
        "batch_size": 10,
        "epochs": 100
    }

    simulation.sim_list([main_template])

    # layers_lst = list(range(1, 6))
    # neurons_lst = list(range(1, 6))
    #
    # def create_sim(layers, neurons):
    #     template = main_template.copy()
    #     template["layers"] = layers
    #     template["neurons"] = neurons
    #     template["name"] = f"L:{layers} N:{neurons}"
    #     simulation.add_from_template(template)

    # simulation.grid_search(
    #     create_sim,
    #     [layers_lst, neurons_lst],
    #     "leafcount", "random_state",
    #     sorted_count=10,
    #     plot_enabled=False
    # )


def prepare(df, colnames, apply_scaling=False):
    X = df.copy()
    X = drop(X, colnames.target_col)
    X = fillna(X, colnames.fillna_cols)
    X = impute(X, colnames.impute_cols)
    X = label_encode(X, colnames.label_encode_cols)
    X = one_hot_encode(X, colnames.onehot_encode_cols)
    X = drop(X, colnames.drop_cols)
    if apply_scaling:
        X = scale(X)
    # print(X)
    # import sys
    # sys.exit()
    y = df[colnames.target_col]
    return X, y


def prepare_for_trees(X):
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = ["Sex", "Cabin", "Embarked", "Ticket"]
    colnames.onehot_encode_cols = []
    colnames.drop_cols = ["Name", "PassengerId"]
    return prepare(df, colnames)


def prepare_for_nn(X):
    colnames = ColNames()
    colnames.target_col = "Survived"
    colnames.fillna_cols = ["Cabin", "Embarked"]
    colnames.impute_cols = ["Age", "Fare"]
    colnames.label_encode_cols = []
    colnames.onehot_encode_cols = ["Sex", "Embarked"]
    colnames.drop_cols = ["Name", "PassengerId", "Ticket", "Cabin"]
    return prepare(df, colnames, apply_scaling=True)


if __name__ == "__main__":

    pd.set_option('display.max_columns', 15)

    # loading CSV
    df = pd.read_csv("train.csv")

    # preparing data
    X, y = prepare_for_nn(df)
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    data_split = DataSplit(train_X, val_X, train_y, val_y)

    # simulate(data_split)
    simulate_nn(data_split)
