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
import itertools


simulations = None
global_data = None
network_id = 0


class GlobalSettings:
    float_digits = 10


class GlobalData:
    data_split = None
    prop_list = []
    prop_aliases = []


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
    losses = [sim.val_loss for sim in simulations]
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
    # list of properties to print
    prop_lst = ["name", "train_accuracy", "val_accuracy", "overfitting"]
    prop_aliases = ["name", "ta", "va", "of"]
    # creating simulations
    create_grid(lists, f)
    # running simulations
    run_all(prop_lst, prop_aliases, jobs=None)
    # printing results
    simulations_copy = simulations.copy()
    simulations_copy.sort(key=lambda x: x.val_loss)
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
        self.train_loss = None
        self.val_losss = None
        self.train_accuracy = None
        self.val_accuracy = None
        self.overfitting = None
        self.leafcount = None
        self.template = None

    def fit_tree(self):
        self.model.fit(
            X=global_data.data_split.train_X,
            y=global_data.data_split.train_y
        )

    def run(self):
        # decision tree
        if self.template["type"] == "dt":
            self.model = DecisionTreeClassifier(
                max_depth=self.template["leafcount"],
                random_state=self.template["random_state"]
            )
            self.fit_tree()
        # random forest
        elif self.template["type"] == "rf":
            self.model = RandomForestClassifier(
                n_estimators=self.template["count"],
                max_depth=self.template["leafcount"],
                random_state=self.template["random_state"]
            )
            self.fit_tree()
        # neural network
        elif self.template["type"] == "nn":
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
            # running model
            self.model.fit(
                global_data.data_split.train_X,
                global_data.data_split.train_y,
                epochs=self.template["epochs"],
                batch_size=self.template["batch_size"],
                verbose=0
            )
        else:
            raise ValueError(f"Unknown type: {self.type}")
        # measuring loss and accuracy
        train_X = global_data.data_split.train_X
        val_X = global_data.data_split.val_X
        train_y = global_data.data_split.train_y
        val_y = global_data.data_split.val_y
        train_predict = np.round(self.model.predict(train_X))
        val_predict = np.round(self.model.predict(val_X))
        self.train_loss = mean_absolute_error(train_y, train_predict)
        self.val_loss = mean_absolute_error(val_y, val_predict)
        self.train_accuracy = accuracy_score(train_y, train_predict)
        self.val_accuracy = accuracy_score(val_y, val_predict)
        self.overfitting = self.val_loss - self.train_loss
        # saving
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
