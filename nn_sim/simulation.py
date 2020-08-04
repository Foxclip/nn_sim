from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from nn_sim.dataset import CPU_MODE
import os
import keras
import numpy as np
import shutil
import time
import multiprocessing
import sys
from nn_sim import plot
import enum
import itertools


simulations = None
global_data = None
network_id = 0


class GlobalSettings:
    pass


class GlobalData:
    full_data = None
    data_split = None
    folds = None
    prop_list = []
    prop_aliases = []
    model_settings = None
    cross_validate = None


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
        return keras.layers.Dense(
            units=self.units,
            input_dim=self.input_dim,
            activation=self.activation
        )


class LossHistory(keras.callbacks.Callback):
    """Stores loss history."""
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))


def init():
    global simulations, global_settings, global_data
    simulations = []
    global_data = GlobalData()
    global_settings = GlobalSettings()


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
    # measuring time
    time1 = time.time()
    # running simulations
    global simulations
    if (jobs is None or jobs > 1) and len(simulations) > 1 and CPU_MODE:
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
    validation = global_data.model_settings.validation
    losses = None
    if validation == ValidationTypes.cross_val:
        losses = [sim.cv_loss for sim in simulations]
    elif validation == ValidationTypes.val_split:
        losses = [sim.val_loss for sim in simulations]
    elif validation == ValidationTypes.none:
        losses = [sim.train_loss for sim in simulations]
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


def grid_search(f, lists, xlabel, ylabel, sorted_count=0, plot_enabled=True):
    # selecting list of properties to print
    prop_lst = None
    prop_aliases = None
    if global_data.model_settings.validation == ValidationTypes.cross_val:
        if global_data.model_settings.task_type == TaskTypes.regression:
            prop_lst = ["name", "cv_loss", "train_loss", "overfitting",
                        "lowest_loss_point"]
            prop_aliases = ["name", "cvl", "tl", "of", "llp"]
        else:
            prop_lst = ["name", "cv_acc", "train_acc", "overfitting",
                        "lowest_loss_point"]
            prop_aliases = ["name", "cva", "ta", "of", "llp"]
    elif global_data.model_settings.validation == ValidationTypes.val_split:
        if global_data.model_settings.task_type == TaskTypes.regression:
            prop_lst = ["name", "train_loss", "val_loss", "overfitting",
                        "lowest_loss_point"]
            prop_aliases = ["name", "tl", "vl", "of", "llp"]
        else:
            prop_lst = ["name", "train_acc", "val_acc", "overfitting",
                        "lowest_loss_point"]
            prop_aliases = ["name", "ta", "va", "of", "llp"]
    elif global_data.model_settings.validation == ValidationTypes.none:
        if global_data.model_settings.task_type == TaskTypes.regression:
            prop_lst = ["name", "train_loss", "lowest_loss_point"]
            prop_aliases = ["name", "tl", "llp"]
        else:
            prop_lst = ["name", "train_acc", "lowest_loss_point"]
            prop_aliases = ["name", "ta", "llp"]
    # creating simulations
    create_grid(lists, f)
    # running simulations
    run_all(prop_lst, prop_aliases, jobs=None)
    # printing results
    simulations_copy = simulations.copy()
    validation = global_data.model_settings.validation
    if validation == ValidationTypes.cross_val:
        simulations_copy.sort(key=lambda x: x.cv_loss)
    elif validation == ValidationTypes.val_split:
        simulations_copy.sort(key=lambda x: x.val_loss)
    elif validation == ValidationTypes.none:
        simulations_copy.sort(key=lambda x: x.train_loss)
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
        if self.template["loss"] == "bce":
            self.template["loss"] = keras.losses.binary_crossentropy
        elif self.template["loss"] == "cce":
            self.template["loss"] = keras.losses.categorical_crossentropy
        # creating model
        self.model = keras.models.Sequential()
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
            history = LossHistory()
            process_name = multiprocessing.current_process().name
            filepath = f"tmp/{process_name}/checkpoint"
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor="loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                period=1
            )
            self.model.fit(
                X,
                y,
                epochs=self.template["epochs"],
                batch_size=self.template["batch_size"],
                verbose=0,
                callbacks=[history, model_checkpoint]
            )
            self.model.load_weights(filepath)
            self.loss_history = history.losses
        else:
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

        target_col = global_data.model_settings.target_col
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
        train_X = None
        train_y = None
        val_X = None
        val_y = None
        if validation == ValidationTypes.val_split:
            train_X = global_data.data_split.train_X
            train_y = global_data.data_split.train_y
            val_X = global_data.data_split.val_X
            val_y = global_data.data_split.val_y
        else:
            train_X = global_data.full_data.drop([target_col], axis=1)
            train_y = global_data.full_data[target_col]
            val_X = train_X
            val_y = train_y

        # training model
        self.run_model(train_X, train_y)

        # calculating loss
        self.train_loss = self.model_loss(train_X, train_y)
        if validation == ValidationTypes.cross_val:
            self.overfitting = self.cv_loss - self.train_loss
            if task_type != TaskTypes.regression:
                self.train_acc = self.model_acc(train_X, train_y)
        elif validation == ValidationTypes.val_split:
            self.val_loss = self.model_loss(val_X, val_y)
            self.overfitting = self.val_loss - self.train_loss
            if task_type != TaskTypes.regression:
                self.train_acc = self.model_acc(train_X, train_y)
                self.val_acc = self.model_acc(val_X, val_y)
        elif validation == ValidationTypes.none:
            if task_type != TaskTypes.regression:
                self.train_acc = self.model_acc(train_X, train_y)

        # finding point of the lowest loss
        lowest_loss_i = np.argmin(self.loss_history)
        self.lowest_loss_point = lowest_loss_i / (len(self.loss_history) - 1)

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
            if type(prop_value) in [np.float64, float]:
                result += f"{prop_aliases[i]}:{prop_value:8.5f} "
            else:
                result += f"{prop_aliases[i]}:{prop_value} "
        return result

    def print_props(self, prop_list, prop_aliases):
        print(self.get_prop_str(prop_list, prop_aliases))
