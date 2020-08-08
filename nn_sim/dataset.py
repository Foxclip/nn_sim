from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from nn_sim import simulation
import shutil
import pandas as pd
import os


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


def scale(df, scale_cols=[], exclude_cols=[]):
    """Scales needed columns with a StandardScaler."""
    scalers = {}
    for col in df:
        if scale_cols and (col not in scale_cols):
            continue
        if col in exclude_cols:
            continue
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df, scalers


def swap(df, column_list, old_value, new_value):
    """Sets cells containing old value to new value."""
    def swap_func(cell_value):
        return new_value if cell_value == old_value else cell_value
    for col in column_list:
        df[col] = df[col].map(swap_func)
    return df


def cut_dataset(X, target_col):
    """Cuts dataset into train and test parts. If value in target_col column is
    missing, the rows goes to X_test, otherwise it goes to X_train."""
    X_train = X.dropna(subset=[target_col])
    X_test = X[X[target_col].isnull()]
    X_test = drop(X_test, target_col)
    return X_train, X_test


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


def train_models(data, scalers, model_settings, layers_lst, neurons_lst):
    """Train models and save them."""
    # deleting saved models
    clear_folder("models")
    # training models
    simulation.nn_grid(data, scalers, model_settings, layers_lst, neurons_lst)


def make_predictions(X, scalers):
    """Make predictions with the best model."""
    print("Making predictions...")
    # loading best model
    from keras.models import load_model
    best_model = load_model("best_model")
    # making predictions
    predict = best_model.predict(X)
    target_col = simulation.global_data.model_settings.target_col
    predict = scalers[target_col].inverse_transform(predict)
    return predict
