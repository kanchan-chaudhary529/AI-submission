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

    # Step 1: Declare list of hidden layer widths (1 to 10)
    hidden_layer_width = list(range(1, 11))  # [1, 2, ..., 10]
    
    # Step 2: Initialize successes array to count 100% accuracy runs
    successes = np.zeros(10)  # Shape (10,)
    
    # Step 3: Initialize epochs array to store epochs for each run
    epochs = np.zeros((10, 10))  # Shape (10, 10)
    
    # Step 4: Nested loops over hidden layer sizes and repetitions
    for h_nodes in hidden_layer_width:
        for repetition in range(10):
            # Step 2 (adapted): Create MLP with h_nodes hidden nodes
            xorMLP = MLPClassifier(
                hidden_layer_sizes=(h_nodes,),
                random_state=repetition,  # Set random_state to repetition index
                solver='lbfgs',  # Common for XOR, ensures convergence
                max_iter=1000,  # Sufficient iterations
                activation='logistic'  # Common for XOR
            )
            
            # Step 3 (adapted): Fit the model to training data
            xorMLP.fit(train_x, train_y)
            
            # Step 4 (adapted): Measure accuracy
            accuracy = xorMLP.score(train_x, train_y)
            
            # Check if accuracy is 100%
            if accuracy == 1.0:
                # Increment successes for this hidden layer size
                successes[h_nodes-1] += 1
                # Store number of epochs taken
                epochs[h_nodes-1][repetition] = xorMLP.n_iter_
    
    # Step 5: Compute efficiency (mean epochs for successful runs or 1000 if no successes)
    efficiency = np.zeros(10)  # Shape (10,)
    for i in range(10):
        successful_epochs = epochs[i][epochs[i] > 0]  # Non-zero epochs (successful runs)
        if len(successful_epochs) > 0:
            efficiency[i] = np.mean(successful_epochs)  # Mean of successful epochs
        else:
            efficiency[i] = 1000  # No successful runs
    
    # Step 6: Create side-by-side plots (adapted from step 5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots
    
    # Left plot: Success rate vs. hidden layer size
    ax[0].plot(hidden_layer_width, successes/10, marker='o')
    ax[0].set_xlabel('Hidden Layer Size')
    ax[0].set_ylabel('Success Rate')
    ax[0].set_title('Success Rate vs. Hidden Layer Size')
    ax[0].grid(True)
    
    # Right plot: Efficiency vs. hidden layer size
    ax[1].plot(hidden_layer_width, efficiency, marker='o')
    ax[1].set_xlabel('Hidden Layer Size')
    ax[1].set_ylabel('Mean Epochs (1000 if no success)')
    ax[1].set_title('Efficiency vs. Hidden Layer Size')
    ax[1].grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
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
        self.data_x = np.genfromtxt(datafilename, delimiter=',')  # Load features
        self.data_y = np.genfromtxt(labelfilename, delimiter=',', dtype=int)  # Load integer labels
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes
 
           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here

        # Step 1: Stratified 70:30 train-test split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, stratify=self.data_y, random_state=12345
        )
        
        # Step 2: Normalize features to [0, 1] using MinMaxScaler
        scaler = MinMaxScaler()
        self.train_x = scaler.fit_transform(self.train_x)  # Fit and transform training data
        self.test_x = scaler.transform(self.test_x)  # Transform test data
        
        # Step 3: Create one-hot encoded labels for MLP if 3+ classes
        if len(np.unique(self.data_y)) >= 3:
            binarizer = LabelBinarizer()
            self.train_y_onehot = binarizer.fit_transform(self.train_y)
            self.test_y_onehot = binarizer.transform(self.test_y)
        else:
            self.train_y_onehot = self.train_y  # Use original labels for binary classification
            self.test_y_onehot = self.test_y        
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
        # KNN: Try k = [1, 3, 5, 7, 9]
        for k in [1, 3, 5, 7, 9]:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x, self.train_y)
            accuracy = model.score(self.test_x, self.test_y)
            self.stored_models["KNN"].append(model)
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1
        
        # Decision Tree: Try all combinations of max_depth, min_samples_split, min_samples_leaf
        for max_depth in [1, 3, 5]:
            for min_split in [2, 5, 10]:
                for min_leaf in [1, 5, 10]:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y)
                    accuracy = model.score(self.test_x, self.test_y)
                    self.stored_models["DecisionTree"].append(model)
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1
        
        # MLP: Try all combinations of hidden_layer_sizes and activation
        for nodes1 in [2, 5, 10]:
            for nodes2 in [0, 2, 5]:
                hidden_layers = (nodes1,) if nodes2 == 0 else (nodes1, nodes2)
                for activation in ['logistic', 'relu']:
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        random_state=12345,
                        max_iter=1000,
                        solver='adam'
                    )
                    model.fit(self.train_x, self.train_y_onehot if len(np.unique(self.data_y)) >= 3 else self.train_y)
                    accuracy = model.score(self.test_x, self.test_y_onehot if len(np.unique(self.data_y)) >= 3 else self.test_y)
                    self.stored_models["MLP"].append(model)
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
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
        # Find the algorithm with the highest accuracy
        best_algorithm = max(self.best_accuracy, key=self.best_accuracy.get)
        best_accuracy = self.best_accuracy[best_algorithm]
        best_index = self.best_model_index[best_algorithm]
        best_model = self.stored_models[best_algorithm][best_index]
        
        return best_accuracy, best_algorithm, best_model
        # <==== insert your code above here
