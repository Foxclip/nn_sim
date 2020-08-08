from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
import os
import numpy as np
import shutil
import time
import multiprocessing
import sys
from nn_sim import plot
import enum
import itertools
import copy
import math


simulations = None
global_data = None
network_id = 0


class GlobalSettings:
    gpu = None
    tensorflow_messages = None


class GlobalData:
    full_data = None
    target_col = None
    folds = None
    prop_list = []
    prop_aliases = []
    model_settings = None
    scalers = None


class TaskTypes(enum.Enum):
    """Types of tasks for neural networks."""
    regression = 0,
    binary_classification = 1,
    multiclass_classification = 2


class ValidationTypes(enum.Enum):
    "Types of model validation"
    none = 0,
    val_split = 1,
    cross_val = 2


class DataSplit:
    """Stores training and validation data that is passed to the models."""
    def __init__(self, train_X, val_X, train_y, val_y):
        self.train_X = train_X
        self.val_X = val_X
        self.train_y = train_y
        self.val_y = val_y


class ModelSettings:
    """Stores settings of model."""
    def __init__(self):
        self.folds = 10


class NeuralNetworkSettings(ModelSettings):
    """Stores settings of a neural network."""
    def __init__(self):
        self.task_type = TaskTypes.binary_classification
        self.intermediate_activations = "relu"
        self.output_count = 1
        self.optimizer = "Adam"
        self.batch_size = 32
        self.epochs = 100
        self.unscale_loss = True


class Dense:
    """Wrapper for keras.layers.Dense."""
    def __init__(self, units, input_dim=None, activation=None):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation

    def __repr__(self):
        input_dim_str = str(self.input_dim) if self.input_dim else "0"
        return f"Dense(u:{self.units}, i:{input_dim_str}, a:{self.activation})"

    def create(self):
        """Converts wrapper to an actual keras layer."""
        from keras.layers import Dense
        return Dense(
            units=self.units,
            input_dim=self.input_dim,
            activation=self.activation
        )


def init():
    global simulations, global_settings, global_data
    simulations = []
    global_data = GlobalData()
    global_settings = GlobalSettings()


def setup():
    # switching between CPU and GPU
    if not global_settings.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # disabling tensorflow debug messages
    if not global_settings.tensorflow_messages:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def add_from_template(template):
    """Creating simulation from a simulation template"""
    new_sim = Simulation()
    # setting simulation properties from template
    new_sim.template = template
    global network_id
    new_sim.id = network_id
    network_id += 1
    new_sim.name = template["name"]
    simulations.append(new_sim)


def _global_init(p_global_settings, p_global_data):
    print(f"Starting process {multiprocessing.current_process().name}")
    init()
    global global_settings, global_data
    global_settings = p_global_settings
    global_data = p_global_data
    setup()


def _run_simulation(sim):
    sim.run()
    if global_data.prop_list:
        sim.print_props(global_data.prop_list, global_data.prop_aliases)
    return sim


def run_all(p_prop_list=[], p_prop_aliases=[], jobs=None):
    print("Running simulations")
    # setting prop list so it will be copied between processes
    global_data.prop_list = p_prop_list
    global_data.prop_aliases = p_prop_aliases
    # doing setup based on global settings
    setup()
    # measuring time
    time1 = time.time()
    # running simulations
    global simulations
    gpu = global_settings.gpu
    if (jobs is None or jobs > 1) and len(simulations) > 1 and not gpu:
        # run multiple processes
        if len(simulations) < 4:
            jobs = len(simulations)
        with multiprocessing.Pool(
            jobs,
            _global_init,
            (global_settings, global_data)
        ) as pool:
            simulations = pool.map(_run_simulation, simulations)
    else:
        # run single process
        for sim in simulations:
            _run_simulation(sim)
    # choosing and saving best model
    losses = [sim.main_loss for sim in simulations]
    min_id = np.argmin(losses)
    if os.path.exists("best_model"):
        shutil.rmtree("best_model")
    shutil.copytree(f"models/{min_id}", "best_model")
    # measuring time
    time2 = time.time()
    time_passed = time2 - time1
    print(f"Time: {time_passed}s")


def create_grid(lists, f):
    if len(lists) <= 1:
        raise ValueError("Should be at least two lists")
    elif len(lists) == 2:
        for l2 in lists[1]:
            for l1 in lists[0]:
                f(l1, l2)
    else:
        for combination in itertools.product(*lists):
            f(*combination)


def sim_list(template_list, plotting=["loss"]):
    # creating simulations
    for template in template_list:
        add_from_template(template)
    # running simulations
    run_all(["name", "loss", "accuracy"], jobs=None)
    # plotting results
    if "--noplot" not in sys.argv:
        if "loss" in plotting:
            plot.plot_loss()


def get_folds(df, foldcount):
    # preparing data
    kfold = None
    folds = None
    target_col = global_data.target_col
    ms = global_data.model_settings
    if ms.task_type != TaskTypes.regression:
        kfold = StratifiedKFold(n_splits=foldcount, shuffle=True,
                                random_state=7)
        folds = list(kfold.split(df, df[target_col]))
    else:
        kfold = KFold(n_splits=foldcount, shuffle=True, random_state=7)
        folds = list(kfold.split(df))
    return folds


def nn_grid(data, target_col, scalers, model_settings, layers_lst,
            neurons_lst):
    """Creates grid of simulations with neural networks and runs them."""

    # starting up
    init()
    # loading data to simulation module
    gd = global_data
    gd.full_data = data
    gd.target_col = target_col
    gd.model_settings = model_settings
    if model_settings.validation == ValidationTypes.cross_val:
        gd.folds = get_folds(data, model_settings.folds)
    gd.scalers = scalers
    gs = global_settings
    gs.gpu = model_settings.gpu
    gs.tensorflow_messages = False

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
            Dense(
                units=-1,  # should be overriden later
                input_dim=gd.full_data.shape[1] - 1,
                activation=model_settings.intermediate_activations
            ),
            Dense(
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

        # creating layers
        for i in range(layer_count - 1):
            new_layer = Dense(units=neuron_count, activation="relu")
            template["layers"].insert(1, new_layer)

        # making pretty template name
        layer_cnt_digits = int(math.log10(max(layers_lst))) + 1
        neuron_cnt_digits = int(math.log10(max(neurons_lst))) + 1
        layer_format = f"{{:{layer_cnt_digits}.0f}}"
        neuron_format = f"{{:{neuron_cnt_digits}.0f}}"
        layer_count_str = layer_format.format(layer_count)
        neuron_count_str = neuron_format.format(neuron_count)
        template["name"] = f"HL:{layer_count_str} N:{neuron_count_str}"

        add_from_template(template)

    grid_search(
        create_sim,
        [layers_lst, neurons_lst],
        "layers", "neurons",
        sorted_count=10,
        plot_enabled=False
    )


def grid_search(f, lists, xlabel, ylabel, sorted_count=0, plot_enabled=True):
    # selecting what properties properties to print based on validation and
    # tak type
    prop_lst = None
    prop_aliases = None
    validation = global_data.model_settings.validation
    task_type = global_data.model_settings.task_type
    no_val = validation == ValidationTypes.none
    val_split = validation == ValidationTypes.val_split
    cross_val = validation == ValidationTypes.cross_val
    regression = task_type == TaskTypes.regression
    if cross_val and regression:
        prop_lst = ["name", "train_loss", "cv_loss", "overfitting",
                    "lowest_loss_point"]
        prop_aliases = ["name", "tl", "cl", "of", "tllp"]
    if cross_val and not regression:
        prop_lst = ["name", "train_acc", "cv_acc", "train_loss", "cv_loss",
                    "overfitting", "lowest_loss_point"]
        prop_aliases = ["name", "ta", "ca", "tl", "cl", "of", "tllp"]
    if val_split and regression:
        prop_lst = ["name", "train_loss", "val_loss", "overfitting",
                    "lowest_loss_point"]
        prop_aliases = ["name", "tl", "vl", "of", "vllp"]
    if val_split and not regression:
        prop_lst = ["name", "train_acc", "val_acc", "train_loss",
                    "val_loss", "overfitting", "lowest_loss_point"]
        prop_aliases = ["name", "ta", "va", "tl", "vl", "of", "vllp"]
    if no_val and regression:
        prop_lst = ["name", "train_loss", "lowest_loss_point"]
        prop_aliases = ["name", "tl", "tllp"]
    if no_val and not regression:
        prop_lst = ["name", "train_acc", "train_loss", "lowest_loss_point"]
        prop_aliases = ["name", "ta", "tl", "tllp"]
    # creating simulations
    create_grid(lists, f)
    # running simulations
    run_all(prop_lst, prop_aliases, jobs=None)
    # printing results
    simulations_copy = simulations.copy()
    simulations_copy.sort(key=lambda x: x.main_loss)
    print("==============================================")
    for sim in simulations_copy[:sorted_count]:
        sim.print_props(prop_lst, prop_aliases)
    open("output.txt", "w")
    file = open("output.txt", "a")
    for sim_i in range(len(simulations_copy)):
        sim = simulations_copy[sim_i]
        file.write(
            f"<{sim_i + 1}> "
            f"{sim.get_prop_str(prop_lst, prop_aliases)}"
        )
        file.write("\n")
    file.close()
    # plotting results if there are two parameter lists
    if "--noplot" not in sys.argv and plot_enabled and len(lists) == 2:
        plot.loss_surface_plot(lists[0], lists[1],
                               xlabel=xlabel, ylabel=ylabel)


class Simulation:
    def __init__(self):
        self.name = "Untitled"
        self.model = None
        self.cv_loss = None
        self.cv_acc = None
        self.train_loss = None
        self.train_acc = None
        self.val_loss = None
        self.val_acc = None
        self.leafcount = None
        self.template = None
        self.main_loss = None

    def create_decision_tree(self):
        self.model = DecisionTreeClassifier(
            max_depth=self.template["leafcount"],
            random_state=self.template["random_state"]
        )

    def create_random_forest(self):
        self.model = RandomForestClassifier(
            n_estimators=self.template["count"],
            max_depth=self.template["leafcount"],
            random_state=self.template["random_state"]
        )

    def create_neural_network(self):
        # choosing the loss function from its short name
        from keras.losses import binary_crossentropy, categorical_crossentropy
        if self.template["loss"] == "bce":
            self.template["loss"] = binary_crossentropy
        elif self.template["loss"] == "cce":
            self.template["loss"] = categorical_crossentropy
        # creating model
        from keras.models import Sequential
        self.model = Sequential()
        for layer in self.template["layers"]:
            self.model.add(layer.create())
        self.model.compile(
            optimizer=self.template["optimizer"],
            loss=self.template["loss"],
            metrics=["acc"]
        )

    def create_model(self):
        if self.template["type"] == "dt":
            self.create_decision_tree()
        elif self.template["type"] == "rf":
            self.create_random_forest()
        elif self.template["type"] == "nn":
            self.create_neural_network()
        else:
            raise ValueError(f"Unknown type: {self.type}")

    def run_model(self, X, y):

        if self.template["type"] == "nn":

            # shortened names
            ms = global_data.model_settings
            binary = ms.task_type != TaskTypes.regression
            val_split = ms.validation == ValidationTypes.val_split
            monitor = "val_loss" if val_split else "loss"

            # setting up model checkpoint
            callbacks = []
            if ms.checkpoint:
                process_name = multiprocessing.current_process().name
                filepath = f"tmp/{process_name}/checkpoint"
                from keras.callbacks import ModelCheckpoint
                model_checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    monitor=monitor,
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=True,
                    mode="auto",
                    period=1
                )
                callbacks.append(model_checkpoint)

            # validation split ratio
            split_ratio = 0.33 if val_split else 0.0

            # training model
            history = self.model.fit(
                X,
                y,
                epochs=self.template["epochs"],
                batch_size=self.template["batch_size"],
                validation_split=split_ratio,
                shuffle=True,
                verbose=0,
                callbacks=callbacks
            )
            history = history.history

            # loading best weights
            if ms.checkpoint:
                self.model.load_weights(filepath)

            # calculating loss and accuracy
            best_epoch = np.argmin(history[monitor])
            self.train_loss = history["loss"][best_epoch]
            if val_split:
                self.val_loss = history["val_loss"][best_epoch]
            if binary:
                self.train_acc = history["acc"][best_epoch]
            if val_split and binary:
                self.val_acc = history["val_acc"][best_epoch]
            self.lowest_loss_point = best_epoch

        else:

            # traning tree
            self.model.fit(X, y)

    def model_loss(self, val_X, val_y):
        val_predict = self.model.predict(val_X)
        score = mean_absolute_error(val_y, val_predict)
        return score

    def model_acc(self, val_X, val_y):
        val_predict = np.round(self.model.predict(val_X))
        score = accuracy_score(val_y, val_predict)
        return score

    def run(self):

        target_col = global_data.target_col
        validation = global_data.model_settings.validation
        task_type = global_data.model_settings.task_type

        # cross validation
        if validation == ValidationTypes.cross_val:
            cv_losses = []
            cv_accs = []
            for train_indices, val_indices in global_data.folds:
                # creating models
                self.create_model()
                # measuring loss and accuracy
                train_data = global_data.full_data.iloc[train_indices, :]
                val_data = global_data.full_data.iloc[val_indices, :]
                train_X = train_data.drop([target_col], axis=1)
                train_y = train_data[target_col]
                val_X = val_data.drop([target_col], axis=1)
                val_y = val_data[target_col]
                self.run_model(train_X, train_y)
                cv_loss = self.model_loss(val_X, val_y)
                if task_type != TaskTypes.regression:
                    cv_acc = self.model_acc(val_X, val_y)
                    cv_accs.append(cv_acc)
                cv_losses.append(cv_loss)
                task_type = task_type
            # final cv score
            self.cv_loss = np.mean(cv_losses)
            if task_type != TaskTypes.regression:
                self.cv_acc = np.mean(cv_accs)

        # creating final model
        self.create_model()

        # selecting data
        train_X = global_data.full_data.drop([target_col], axis=1)
        train_y = global_data.full_data[target_col]

        # training model
        self.run_model(train_X, train_y)

        # setting main loss
        if validation == ValidationTypes.cross_val:
            self.main_loss = self.cv_loss
        elif validation == ValidationTypes.val_split:
            self.main_loss = self.val_loss
        elif validation == ValidationTypes.none:
            self.main_loss = self.train_loss

        # calculating overfitting measure
        if validation != ValidationTypes.none:
            self.overfitting = self.main_loss - self.train_loss
        # saving model
        self.model.save(f"models/{self.id}")
        # this is needed to avoid sending model back to main thread, which
        # causes error, since keras model cannot be pickled
        self.model = None

    def get_prop_str(self, prop_list, prop_aliases):
        result = ""
        for i, prop_name in enumerate(prop_list):
            prop_value = getattr(self, prop_name)
            if prop_name == "name":
                result += f"{prop_value} "
                continue
            gd = global_data
            ms = gd.model_settings
            loss_properties = ["train_loss", "val_loss", "cv_loss"]
            if ms.unscale_loss and prop_name in loss_properties:
                scaler = gd.scalers[gd.target_col]
                prop_value = scaler.inverse_transform([prop_value])[0]
            if type(prop_value) in [np.float64, float]:
                result += f"{prop_aliases[i]}:{prop_value:8.5f} "
            else:
                result += f"{prop_aliases[i]}:{prop_value} "
        return result

    def print_props(self, prop_list, prop_aliases):
        print(self.get_prop_str(prop_list, prop_aliases))
