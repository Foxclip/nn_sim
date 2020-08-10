from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from nn_sim import simulation
import shutil
import pandas as pd
import os


loaded_dataset = None
target_col = None
scalers = None
X_train = None
X_test = None


def intersect_columns(column_list):
    return [col for col in column_list if col in loaded_dataset.columns]


def fillna(column_list, value="missing"):
    """Fills missing values with 'missing'"""
    column_list = intersect_columns(column_list)
    for col in column_list:
        loaded_dataset[col].fillna(value, inplace=True)


def drop(column_list):
    """Drops columns from the dataframe."""
    column_list = intersect_columns(column_list)
    loaded_dataset.drop(column_list, axis=1, inplace=True)


def impute(column_list):
    """Imputes needed columns."""
    column_list = intersect_columns(column_list)
    for col in column_list:
        type_object = loaded_dataset[col].dtype == np.object
        strategy = "most_frequent" if type_object else "mean"
        imputer = SimpleImputer(strategy=strategy)
        loaded_dataset[col] = imputer.fit_transform(loaded_dataset[[col]])


def label_encode(column_list):
    """Label encodes needed columns."""
    column_list = intersect_columns(column_list)
    for col in column_list:
        loaded_dataset[col] = LabelEncoder().fit_transform(loaded_dataset[col])


def one_hot_encode(column_list):
    """One-hot encodes needed columns."""
    column_list = intersect_columns(column_list)
    global loaded_dataset
    if not column_list:
        return
    # converting to column to string type
    for col in column_list:
        loaded_dataset[col] = loaded_dataset[col].astype(str)
    # creating encoded columns
    encoder = OneHotEncoder()
    data = loaded_dataset[column_list]
    encoded = pd.DataFrame(encoder.fit_transform(data).toarray())
    # creating columns titles
    categories = encoder.categories_
    for col_i in range(len(column_list)):
        categories[col_i] = column_list[col_i] + " " + categories[col_i]
    categories = np.concatenate(categories)
    encoded.columns = categories
    # gluing dataset with encoded part
    not_encoded = loaded_dataset.drop(column_list, axis=1)
    loaded_dataset = pd.concat([not_encoded, encoded], axis=1)


def scale(scale_cols=[], exclude_cols=[]):
    """Scales needed columns with a StandardScaler."""
    scale_cols = intersect_columns(scale_cols)
    global scalers
    scalers = {}
    for col in loaded_dataset:
        if scale_cols and (col not in scale_cols):
            continue
        if col in exclude_cols:
            continue
        scaler = StandardScaler()
        loaded_dataset[col] = scaler.fit_transform(loaded_dataset[[col]])
        scalers[col] = scaler


def swap(column_list, old_value, new_value):
    """Sets cells containing old value to new value."""
    column_list = intersect_columns(column_list)
    def swap_func(cell_value):  # noqa
        return new_value if cell_value == old_value else cell_value
    for col in column_list:
        loaded_dataset[col] = loaded_dataset[col].map(swap_func)


def leave_columns(column_list):
    """Leaves only specified columns."""
    column_list = intersect_columns(column_list)
    global loaded_dataset
    loaded_dataset = loaded_dataset[column_list]


def cut_dataset():
    """Cuts dataset into train and test parts. If value in target_col column is
    missing, the rows goes to X_test, otherwise it goes to X_train."""
    global X_train, X_test
    X_train = loaded_dataset.dropna(subset=[target_col])
    X_test = loaded_dataset[loaded_dataset[target_col].isnull()]
    X_test = X_test.drop([target_col], axis=1)


def clear_folder(folder):
    """Deletes contents of a folder."""
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def train_models(model_settings, layers_lst, neurons_lst):
    """Train models and save them."""
    # cutting dataset to train and test parts
    cut_dataset()
    # deleting saved models
    clear_folder("models")
    # training models
    simulation.nn_grid(X_train, target_col, scalers, model_settings,
                       layers_lst, neurons_lst)


def make_predictions():
    """Make predictions with the best model."""
    print("Making predictions...")
    # loading best model
    from keras.models import load_model
    best_model = load_model("best_model")
    # making predictions
    predict = best_model.predict(X_test)
    predict = scalers[target_col].inverse_transform(predict)
    return predict


def load_dataset(df):
    global loaded_dataset
    loaded_dataset = df.copy()
