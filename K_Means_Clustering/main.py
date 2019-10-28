import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load data in Numpy format
X = np.load('Data/data.npy')

## FInding no. of samples
data_len = X.shape[0]

## Init the distance_matrix which will contain distance between every sample.
data_distance_matrix = np.zeros((data_len, data_len))

def make_data_distance_matrix():
    '''Compute distance between every sample pair in Dataset and stores globally.

    Args:
        None

    Returns:
        None, stores result in data_distance_matrix
    '''
    for i in range(data_len):
        for j in range(data_len):
            diff = X[i] - X[j]
            distance = np.linalg.norm(diff)
            data_distance_matrix[i][j] = distance

def init_centres_random(k):
    '''Random intialization of centroids usin np.random.

    Args:
        k - Number of clusters

    Returns:
        indices of each centroid.
    '''
    inds = np.random.choice(data_len, k)
    return inds

def find_maximal_point(inds):
    '''Find ith sample that is maximally placed from previous (i-1) points used
       in Strategy 2.

    Args:
        inds - Indices of each of the (i-1) Centroids

    Returns:
        index - Index of ith sample
    '''
    max = -1
    index = -1
    for i in range(data_len):
        sum = np.sum(data_distance_matrix[inds, i])
        if sum > max:
            if i not in inds:
                max = sum
                index = i
    return index

def init_centres_max(k):
    '''Implementation of Strategy 2 Initialization

    Args:
        k - No. of cluster centres

    Returns:
        centroid_index_list - Indices of each centroid
    '''
    centroid_index_list = []
    centroid_index_list.append(np.random.choice(data_len, 1)[0])
    for c in range(2, k+1):
        centroid_index_list.append(find_maximal_point(centroid_index_list))
    return centroid_index_list


def calculate_distances(centroids):
    '''Calculate distance of every sample from each centroid

    Args:
        centroids - Array of centroid values

    Returns:
        dist_matrix - (data_len x k) matrix of distances
    '''
    dist_matrix = []
    for obs in X:
        #print obs, centroids
        dists = np.linalg.norm((obs - centroids), axis=1)
        dist_matrix.append(dists)
    return np.array(dist_matrix)

def assign_clusters(centroids):
    '''Assign each sample in X to a cluster (nearest centroid)

    Args:
        centroids - Array of centroid values

    Returns:
        dist_matrix - Distances of samples and centroids
        clusters - 1D array of cluster assignments (0-k)
    '''
    dist_matrix = calculate_distances(centroids)
    clusters = np.argmin(dist_matrix, axis=1)
    return dist_matrix, clusters

def k_means(k, strategy):
    '''Implementation of K-means algorithm with both initialization strategies.

    Args:
        k - Number of clusters
        strategy - Init strategy; 1 - Random; 2 - Max.

    Returns:
        centroids - Final centroid values
        dist_matrix - distances of samples from centroids
        clusters - Final cluster assignments
        log - variable storing the changes of centroids
    '''
    log = np.zeros((1,2))
    if strategy == 1:
        centroid_inds = init_centres_random(k)
    else:
        centroid_inds = init_centres_max(k)
    centroids = X[centroid_inds]
    dist_matrix, clusters = assign_clusters(centroids)
    ctr=0
    log = np.vstack((log, centroids))
    while True:
        ctr += 1
        new_centroids = np.zeros((k, 2))
        for i in range(k):
            inds = np.where(clusters == i)[0]
            if inds.shape[0] == 0:
                continue
            c = np.sum(X[inds, :], axis=0)
            new_centroids[i] = c / float(inds.shape[0])
        log = np.vstack((log, new_centroids))
        if ctr > 500:
            print centroids - new_centroids
        #print log
        if np.sum(centroids == new_centroids) == k*2:
            break
        else:
            centroids = np.copy(new_centroids)
            dist_matrix, clusters = assign_clusters(centroids)
    return centroids, dist_matrix, clusters, log

def objective_function(k, centroids, clusters):
    '''Compute objective function as described in the report

    Args:
        k - No. of clusters
        centroids - Final cluster centroids
        clusters - Final cluster assignments

    Returns:
        sum - Cost for a given clustering
    '''
    sum = 0.0
    for i in range(k):
        inds = np.where(clusters == i)[0]
        if inds.shape[0] == 0:
            continue
        dists = np.linalg.norm((X[inds, :] - centroids[i]), axis=1)
        sum += np.sum(dists**2)
    return sum

def make_plots(costs, name):
    '''Plot Number of clusters vs objective function for 2 runs at a time

    Args:
        costs - objective function values for number of clusters ranging (2,10)
        for 2 runs at a time
        name - name of the file to store plot png

    Returns:
        None, plots the graphs.
    '''
    plt.plot(range(2,11), costs[:9], label='1st Run')
    plt.scatter(range(2,11), costs[:9])
    plt.plot(range(2,11), costs[9:], label='2nd Run')
    plt.scatter(range(2,11), costs[9:])
    plt.legend()
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Cost")
    plt.savefig(name+'.png')
    plt.show()

def run_k_means(start_seed, stop_seed, strategy):
    '''Call K-means on dataset with seed values and init strategy

    Args:
        start_seed - Seed values for 1st run.
        stop_seed - Seed value for 2nd run.
        strategy - Initialization type;
                        1: Random init
                        2: Maximal init

    Returns:
        costs - objective function values for 2 runs k -> (2, 10)
    '''
    centres = []
    costs = []
    for s in range(start_seed, stop_seed):
        for i in range(2, 11):
            np.random.seed(s)
            c, d, clusters,log = k_means(i,strategy)
            centres.append(c)
            costs.append(objective_function(i, c, clusters))
    print ('Run 1: ')
    print (costs[:9])
    print ('Run 2')
    print (costs[9:])
    return costs

if __name__ == '__main__':

    ## Initialize data_distance_matrix with pairwise distances
    make_data_distance_matrix()

    ## Run K-means with strategy 1 init and plot
    print ('Costs For strategy 1: ')
    costs = run_k_means(5, 7, 1)
    make_plots(costs, 'random')

    ## Run K-means with strategy 2 init and plot
    print ('Costs For strategy 2: ')
    costs = run_k_means(18, 20, 2)
    make_plots(costs, 'Max')
