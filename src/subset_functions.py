import itertools


def subsets(X):
    
    ''' use it for the full computation of shapley values
    computes all the possible subsets of features in the data '''
    
    arr = list(range(X.shape[0]))
    for n_index in range(len(arr)):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)
            
def subsets_bounded(X, bound):
    
    ''' use it for the bounded computation of shapley values
    computes all the possible subsets of the feaatures in the data with size up to the bound '''
    
    arr = list(range(X.shape[0]))
    for n_index in range(bound):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)
            
