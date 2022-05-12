import itertools

# =======================================================================================
# Subset Functions
# =======================================================================================

def build_subsets(X):
    arr = list(range(X.shape[0]))
    for n_index in range(len(arr)):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)
            
def build_subsets_limited(X, limit):
    arr = list(range(X.shape[0]))
    for n_index in range(limit):
        for value in itertools.combinations(arr, n_index + 1):
            yield list(value)
            
