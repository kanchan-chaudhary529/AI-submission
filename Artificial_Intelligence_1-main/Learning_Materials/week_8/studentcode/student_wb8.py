from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to  complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object
    
    ax: matplotlib.axes.Axes
        axis
    """
    
    # ====> insert your code below here
     # Define model sizes
    hidden_layer_width = list(range(1, 11))
   # Array to count successful runs for each width
    successes = np.zeros(10, dtype=int)
    # Array to record epochs for each run
    epochs = np.zeros((10, 10), dtype=int)

    # Run experiments
    for idx, h_nodes in enumerate(hidden_layer_width):
        for repetition in range(10):
            # Configure MLP with reproducible random_state
            xorMLP = MLPClassifier(
                hidden_layer_sizes=(h_nodes,),
                max_iter=1000,
                alpha=1e-4,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=repetition
            )
            xorMLP.fit(train_x, train_y)
            acc = xorMLP.score(train_x, train_y)
            # Record success and epochs
            if acc == 1.0:
                successes[idx] += 1
                epochs[idx, repetition] = xorMLP.n_iter_


    efficiency = np.zeros(10, dtype=float)
    for idx in range(10):
        if successes[idx] == 0:
            efficiency[idx] = 1000
        else:
     
            efficiency[idx] = np.mean(epochs[idx][epochs[idx] > 0])

    # Create plots
    fig, ax = plt.subplots(1, 2)
    # Reliability plot
    ax[0].plot(hidden_layer_width, successes / 10.0)
    ax[0].set_title("Reliability")
    ax[0].set_xlabel("Hidden Layer Width")
    ax[0].set_ylabel("Success Rate")
    # Efficiency plot
    ax[1].plot(hidden_layer_width, efficiency)
    ax[1].set_title("Efficiency")
    ax[1].set_xlabel("Hidden Layer Width")
    ax[1].set_ylabel("Mean Epochs")
    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """ 
    
    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.
        
        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """

        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x = np.genfromtxt(datafilename, delimiter=",")
        self.data_y = np.genfromtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes
 
           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, random_state=12345
        )
        
        #this is Manual standardization using NumPy
        train_mean = np.mean(self.train_x, axis=0)
        train_std = np.std(self.train_x, axis=0)
        train_std[train_std == 0] = 1
        self.train_x = (self.train_x - train_mean) / train_std
        self.test_x = (self.test_x - train_mean) / train_std
        
        # One-hot encode labels for MLP if more than 2 classes
        if len(np.unique(self.data_y)) > 2:
            lb = LabelBinarizer()
            self.train_y_mlp = lb.fit_transform(self.train_y)
            self.test_y_mlp = lb.transform(self.test_y)
        else:
            self.train_y_mlp = self.train_y
            self.test_y_mlp = self.test_y
        # <==== insert your code above here
    
    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.
        
        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and  lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        
        """
        # ====> insert your code below here
        # KNN
        k_values = [1, 3, 5, 7, 9]
        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x, self.train_y)
            acc = model.score(self.test_x, self.test_y) * 100
            self.stored_models["KNN"].append(model)
            if acc > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = acc
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1
        
        # Decision Tree
        max_depth_values = [1, 3, 5]
        min_samples_split_values = [2, 5, 10]
        min_samples_leaf_values = [1, 5, 10]
        for max_depth in max_depth_values:
            for min_samples_split in min_samples_split_values:
                for min_samples_leaf in min_samples_leaf_values:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y)
                    acc = model.score(self.test_x, self.test_y) * 100
                    self.stored_models["DecisionTree"].append(model)
                    if acc > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = acc
                        self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1
        
        # MLP
        first_layer_values = [2, 5, 10]
        second_layer_values = [0, 2, 5]
        activation_values = ["logistic", "relu"]
        for first_layer in first_layer_values:
            for second_layer in second_layer_values:
                for activation in activation_values:
                    hidden_layer_sizes = (first_layer,) if second_layer == 0 else (first_layer, second_layer)
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        max_iter=1000,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y_mlp)
                    acc = model.score(self.test_x, self.test_y_mlp) * 100
                    self.stored_models["MLP"].append(model)
                    if acc > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = acc
                        self.best_model_index["MLP"] = len(self.stored_models["MLP"]) - 1
        # <==== insert your code above here

    
    def report_best(self) :
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"
        
        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        best_alg = max(self.best_accuracy, key=self.best_accuracy.get)
        best_acc = self.best_accuracy[best_alg]
        best_model = self.stored_models[best_alg][self.best_model_index[best_alg]]
        return best_acc, best_alg, best_model
        # <==== insert your code above here
