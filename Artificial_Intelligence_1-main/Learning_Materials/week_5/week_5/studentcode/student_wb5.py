# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use
        
        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot
        
        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here
    # get the data from file into a numpy array
    X = np.genfromtxt(datafile_name, delimiter=',')
    
    # create a K-Means cluster model with  the specified number of clusters
    cluster_model = KMeans(n_clusters=K, n_init=10)
    cluster_model.fit(X)
    cluster_ids = cluster_model.predict(X)
    
    # create a canvas(fig) and axes to hold your visualisation
    num_feat = X.shape[1]
    fig, ax = plt.subplots(num_feat, num_feat, figsize=(12, 12))
    plt.set_cmap('viridis')
    
    colors = plt.cm.viridis(np.linspace(0, 1, K))
    # make the visualisation
    for feature1 in range(num_feat):
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top')
        
        for feature2 in range(num_feat):
            x_data = X[:, feature1]
            y_data = X[:, feature2]
            
            if feature1 != feature2:
                ax[feature1, feature2].scatter(x_data, y_data, c=cluster_ids, s=40)
            else:
                # Create colored histograms
                for cluster in range(K):
                    cluster_data = x_data[cluster_ids == cluster]
                    ax[feature1, feature2].hist(cluster_data, bins=15, 
                                              color=colors[cluster], 
                                              alpha=0.7,
                                              edgecolor='black')
                
    # remember to put your user name into the title as specified
    fig.suptitle(f"Visualisation of {K} clusters by k-chaudhary", fontsize=16, y=0.925)

    # save it to file as specified
    fig.savefig('myVisualisation.jpg')

    # if you don't delete the line below there will be problem!
    
    return fig,ax
    
    # <==== insert your code above here
