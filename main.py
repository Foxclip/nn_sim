from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
    for col_i in range(len(onehot_encode_cols)):
        categories[col_i] = onehot_encode_cols[col_i] + " " + categories[col_i]
    categories = np.concatenate(categories)
    encoded.columns = categories
    df = df.drop(column_list, axis=1)
    df_encoded = pd.concat([df, encoded], axis=1)
    return df_encoded


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


def save_result(model):

    # loading test dataset
    df_test = pd.read_csv("test.csv")
    new_df = df_test[["PassengerId"]]

    # preparing test dataset
    X = df_test.copy()
    X = fillna(X, fillna_cols)
    X = drop(X, drop_cols)
    X = impute(X, impute_cols)
    X = label_encode(X, label_encode_cols)

    # making predictions on test dataset
    pred_test = model.predict(X)

    # making and saving new dataframe
    new_df = new_df.assign(Survived=pred_test)
    print(new_df)
    new_df.to_csv("result.csv", index=False)


def train_and_save(data_split):

    # training model
    # model = DecisionTreeClassifier(max_depth=10, random_state=17)
    model = RandomForestClassifier(n_estimators=100, max_depth=11)
    model.fit(data_split.train_X, data_split.train_y)

    save_result(model)

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


def simulate(data_split):

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


if __name__ == "__main__":

    pd.set_option('display.max_columns', 15)

    # loading CSV
    df = pd.read_csv("train.csv")

    # column names
    target_col = "Survived"
    fillna_cols = ["Cabin", "Embarked"]
    impute_cols = ["Age", "Fare"]
    label_encode_cols = ["Sex", "Cabin", "Embarked", "Ticket"]
    onehot_encode_cols = []
    drop_cols = ["Name", "PassengerId"]

    # preparing data
    X = df.copy()
    X = fillna(X, fillna_cols)
    X = drop(X, target_col)
    X = drop(X, drop_cols)
    X = impute(X, impute_cols)
    X = label_encode(X, label_encode_cols)
    X = one_hot_encode(X, onehot_encode_cols)
    # print(X)
    y = df[target_col]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    data_split = DataSplit(train_X, val_X, train_y, val_y)

    simulate(data_split)
    # train_and_save(data_split)
