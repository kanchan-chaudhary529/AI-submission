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
     model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_x, train_y)

    # Get predicted probabilities for the positive class (assume binary classification)
    prob_pos = model.predict_proba(train_x)[:, 1]

    # Calculate the calibration curve
    prob_true, prob_pred = calibration_curve(train_y, prob_pos, n_bins=10, strategy='uniform')

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker='o', label='KNN')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    ax.set_title("Reliability Plot (Calibration Curve)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability in Bin")
    ax.legend()
    ax.grid(True)
    
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
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x = np.loadtxt(datafilename, delimiter=',')
        self.data_y = np.loadtxt(labelfilename, delimiter=',').astype(int)
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

        self.scaler = StandardScaler()
        self.train_x = self.scaler.fit_transform(self.train_x)
        self.test_x = self.scaler.transform(self.test_x)

        if len(np.unique(self.data_y)) > 2:
            self.train_y_oh = to_categorical(self.train_y)
            self.test_y_oh = to_categorical(self.test_y)
        else:
            self.train_y_oh = self.train_y
            self.test_y_oh = self.test_y

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
        for k in [1, 3, 5, 7]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x, self.train_y)
            acc = model.score(self.test_x, self.test_y)
            self.stored_models["KNN"].append(model)
            if acc > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = acc
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1

        # Decision Tree
        for d in [3, 5, 10, None]:
            model = DecisionTreeClassifier(max_depth=d, random_state=12345)
            model.fit(self.train_x, self.train_y)
            acc = model.score(self.test_x, self.test_y)
            self.stored_models["DecisionTree"].append(model)
            if acc > self.best_accuracy["DecisionTree"]:
                self.best_accuracy["DecisionTree"] = acc
                self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1

        # MLP
        for hidden in [(10,), (20,), (10, 10)]:
            model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000, random_state=12345)
            model.fit(self.train_x, self.train_y_oh)
            acc = model.score(self.test_x, self.test_y_oh)
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
        index = self.best_model_index[best_alg]
        model = self.stored_models[best_alg][index]
        accuracy = self.best_accuracy[best_alg]
        return accuracy, best_alg, model
        # <==== insert your code above here
