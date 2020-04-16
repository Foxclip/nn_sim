# switching between CPU and GPU
CPU_MODE = True
if CPU_MODE:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from keras import Sequential
import keras
import time
import multiprocessing
import sys
import plot
import itertools


simulations = None
global_data = None


class GlobalSettings:
    pass


class GlobalData:
    data_split = None
    prop_list = []


class Dense:
    """Wrapper for keras.layers.Dense."""
    def __init__(self, units, input_dim=None, activation=None):
        self.units = units
        self.input_dim = input_dim
        self.activation = activation

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

    model = None

    # decision tree
    if template["type"] == "dt":
        model = DecisionTreeClassifier(
            max_depth=template["leafcount"],
            random_state=template["random_state"]
        )

    # random forest
    elif template["type"] == "rf":
        model = RandomForestClassifier(
            n_estimators=template["count"],
            max_depth=template["leafcount"],
            random_state=template["random_state"]
        )

    # neural network
    elif template["type"] == "nn":
        model = Sequential()
        for layer in template["layers"]:
            model.add(layer.create())
        model.compile(
            optimizer=template["optimizer"],
            loss=template["loss"],
            metrics=["acc"]
        )

    new_sim = Simulation(model)
    if template["type"] in ["dt", "rf"]:
        new_sim.leafcount = template["leafcount"]
    new_sim.name = template["name"]
    new_sim.type = template["type"]
    new_sim.epochs = template["epochs"]
    new_sim.batch_size = template["batch_size"]
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
        sim.print_props(global_data.prop_list)
    return sim


def run_all(p_prop_list=[], jobs=None):
    print("Running simulations")
    global_data.prop_list = p_prop_list
    time1 = time.time()
    global simulations
    if (jobs is None or jobs > 1) and len(simulations) > 1:
        if len(simulations) < 4:
            jobs = len(simulations)
        with multiprocessing.Pool(
            jobs,
            _global_init,
            (global_settings, global_data)
        ) as pool:
            simulations = pool.map(_run_simulation, simulations)
    else:
        for sim in simulations:
            _run_simulation(sim)
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
    run_all(["name", "loss"], jobs=None)
    # plotting results
    if "--noplot" not in sys.argv:
        if "loss" in plotting:
            plot.plot_loss()


def grid_search(f, lists, xlabel, ylabel, sorted_count=0, plot_enabled=True):
    # creating simulations
    create_grid(lists, f)
    # running simulations
    run_all(["name", "loss"], jobs=None)
    # printing results
    simulations_copy = simulations.copy()
    simulations_copy.sort(key=lambda x: x.loss)
    print("==============================================")
    for sim in simulations_copy[:sorted_count]:
        sim.print_props(["name", "loss"])
    open("output.txt", "w")
    file = open("output.txt", "a")
    for sim_i in range(len(simulations_copy)):
        sim = simulations_copy[sim_i]
        file.write(f"<{sim_i + 1}> {sim.get_prop_str(['name', 'loss'])}")
        file.write("\n")
    file.close()
    # plotting results if there are two parameter lists
    if "--noplot" not in sys.argv and plot_enabled and len(lists) == 2:
        plot.loss_surface_plot(lists[0], lists[1],
                               xlabel=xlabel, ylabel=ylabel)


class Simulation:
    def __init__(self, model):
        self.name = "Untitled"
        self.model = model
        self.loss = None
        self.leafcount = None
        self.epochs = None
        self.batch_size = None

    def run(self):
        if self.type == "nn":
            self.model.fit(
                global_data.data_split.train_X,
                global_data.data_split.train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
        elif self.type in ["dt", "rf"]:
            self.model.fit(
                X=global_data.data_split.train_X,
                y=global_data.data_split.train_y
            )
        else:
            raise ValueError(f"Unknown type: {self.type}")
        predict = self.model.predict(global_data.data_split.val_X)
        self.loss = mean_absolute_error(predict, global_data.data_split.val_y)

    def get_prop_str(self, prop_list):
        result = ""
        for prop_name in prop_list:
            prop_value = getattr(self, prop_name)
            if prop_name == "name":
                result += f"{prop_value} "
                continue
            result += f"{prop_name}:{prop_value} "
        return result

    def print_props(self, prop_list):
        print(self.get_prop_str(prop_list))
