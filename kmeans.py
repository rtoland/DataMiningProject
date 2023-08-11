import pandas as pd
from numpy import array, array_equal, sort, mean
from numpy.linalg import norm
from random import seed, randint

N_CLUSTERS = 2 # Number of clusters
COLS = 8 # Number of columns


def stop(clusters):
    '''
    Determines whether to terminate the scripts.

    Takes in a list of cluster lists: The first cluster is the previous iteration and the second is the current iteration.
    Each list is recursively sorted so that they can compared. If they are equal, then that means that the clusters haven't
    changed since the previous iteration, which means that the terminating conditions have been met.
    '''
    a = [sort(x) for x in clusters[0]]
    b = [sort(y) for y in clusters[1]]
    return array_equal(array(a), array(b))

def calc_distances(centroids, row):
    '''
    Returns a tuple of the distances between a particular row and each centroid
    '''
    return (abs(norm(row - centroids[0])), abs(norm(row - centroids[1])))

seed() # Set random seed

# Initial centroids are random
# TODO: Get actual number of columns and ranges for each column
centroids = [[randint(0, 100) for i in range(8)], [randint(0, 100) for i in range(8)]]
clusters_1 = [[], []] # Previous set of clusters
clusters_2 = [[], []] # Current set of clusters
finished = False # Boolean flag to track completion

df = pd.read_csv('Alaska.zip', compression='zip') # Load dataframe

DF_LEN = len(df) # Number of objects in dataframe

# Initial clustering
for i in range(DF_LEN):
    idx = i # Index of current object
    x = array(list(df.iloc[i,:])) # Convert object to numpy array and assign to 'x' for readability
    distances = calc_distances(centroids, x) # Calculate distances between object and centroids
    # Append the object's index to the cluster corresponding with the minimum distance
    clusters_1[distances.index(min(distances))].append(idx)

centroids = [mean(x) for x in centroids] # Calculate new centroids

#Successive rounds
while not finished:
    for i in range(N_CLUSTERS): # Iterate over each cluster
        l = len(clusters_1[i]) # Number of indices in current cluster
        for j in range(l): # Iterate over each index in current cluster
            idx = clusters_1[i, j] # Index under current examination
            x = array(list(df.iloc[idx])) # Convert object at current index to numpy array and assign to 'x' for readability
            distances = calc_distances(centroids, x) # Calculate distances between object and centroids
            # Append the object's index to the cluster corresponding with the minimum distance
            # This time use the current clusters rather than previous clusters
            clusters_2[distances.index(min(distances))].append(idx)
    centroids = [mean(x) for x in centroids] # Update centroids
    finished = stop([array(clusters_1), array(clusters_2)]) # Check for termination