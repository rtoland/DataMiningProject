import pandas as pd
from numpy import array, array_equal, zeros
from numpy.linalg import norm
from numpy.random import seed, randint
from itertools import chain

N_CLUSTERS = 2 # Number of clusters
COLS = 8 # Number of columns
FILENAME = 'test.csv'


def stop(clusters):
    '''
    Determines whether to terminate the scripts.

    Takes in a list of cluster lists: The first cluster is the previous iteration and the second is the current iteration.
    Each list is recursively sorted so that they can compared. If they are equal, then that means that the clusters haven't
    changed since the previous iteration, which means that the terminating conditions have been met.
    '''
    a = list(chain(*clusters[0])).sort()
    b = list(chain(*clusters[1])).sort()
    return array_equal(array(a), array(b))

def calc_distances(centroids, row):
    '''
    Returns a tuple of the distances between a particular row and each centroid
    '''
    return (abs(norm(row - centroids[0])), abs(norm(row - centroids[1])))

def update_centroids(clusters):
    '''
    Calculate the new centroids.

    Takes in the clusters list and creates two new centroids initialized to zeros. Iterates over each index in each cluster and add the 
    corresponding row from the dataframe. At the end, the new centroids are divided by 
    '''
    new_centroids = [zeros(COLS) for i in range(N_CLUSTERS)] # Two new zeros arrays to start accumulating the sum of each cluster
    lengths = [len(x) for x in clusters] # Number of indices in each cluster
    # For each cluster, iterates over each index and add that to the corresponding accumulator
    for i in range(N_CLUSTERS):
        for j in range(lengths[i]):
            y = array(list(df.iloc[clusters[i][j]]))
            new_centroids[i] += y
    # Returns the accumulators divided by the number of indices corresponding to each cluster to get the mean of each cluster.
    # i.e., centroid
    return [new_centroids[i] / lengths[i] for i in range(N_CLUSTERS)]

seed() # Set random seed

# Initial centroids are random
# TODO: Get actual number of columns and ranges for each column
centroids = [randint(-100, 100, size=(COLS)) for i in range(N_CLUSTERS)]
clusters_1 = [[], []] # Previous set of clusters
clusters_2 = [[], []] # Current set of clusters
finished = False # Boolean flag to track completion

# df = pd.read_csv(FILENAME, compression='zip') # Load dataframe
df = pd.read_csv(FILENAME)

DF_LEN = len(df) # Number of objects in dataframe

# Initial clustering
for i in range(DF_LEN):
    idx = i # Index of current object
    x = array(list(df.iloc[i])) # Convert object to numpy array and assign to 'x' for readability
    distances = calc_distances(centroids, x) # Calculate distances between object and centroids
    # Append the object's index to the cluster corresponding with the minimum distance
    clusters_1[distances.index(min(distances))].append(idx)

centroids = update_centroids(clusters_1) # Calculate new centroids

#Successive rounds
while not finished:
    for i in range(N_CLUSTERS): # Iterate over each cluster
        l = len(clusters_1[i]) # Number of indices in current cluster
        for j in range(l): # Iterate over each index in current cluster
            idx = clusters_1[i][j] # Index under current examination
            x = array(list(df.iloc[idx])) # Convert object at current index to numpy array and assign to 'x' for readability
            distances = calc_distances(centroids, x) # Calculate distances between object and centroids
            # Append the object's index to the cluster corresponding with the minimum distance
            # This time use the current clusters rather than previous clusters
            clusters_2[distances.index(min(distances))].append(idx)
    centroids = update_centroids(clusters_2) # Update centroids
    finished = stop([clusters_1, clusters_2]) # Check for termination
    clusters_1 = clusters_2
    clusters_2 = [[], []]

print(clusters_1)
