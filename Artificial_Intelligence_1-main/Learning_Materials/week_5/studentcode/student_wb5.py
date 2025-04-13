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
     df = pd.read_csv(datafile_name)
    data = df[feature_names].to_numpy()

    # create a K-Means cluster model with  the specified number of clusters
     kmeans = KMeans(n_clusters=K, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data)

    # create a canvas(fig) and axes to hold your visualisation
      fig, ax = plt.subplots(figsize=(8, 6))

    # make the visualisation
    # remember to put your user name into the title as specified
     scatter = ax.scatter(data[:, 0], data[:, 1], c=df['Cluster'], cmap='viridis', edgecolors='k')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("K-Means Clustering Visualization - YourUsername")
    
    # add legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    


    # save it to file as specified
     plt.savefig("cluster_visualization.png")

    # if you don't delete the line below there will be problem!
    raise NotImplementedError("Complete the function")
    
    return fig,ax
    
    # <==== insert your code above here
