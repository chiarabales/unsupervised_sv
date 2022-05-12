import numpy as np
import math
import multiprocessing as mp

from pyitlib import discrete_random_variable as drv

import subset_functions as sf

''' Characteristic Functions '''

def entropy(X):
    # compute the Shannon entropy H of a set of random variables X_1,...,X_n
    return drv.entropy_joint(X)

def total_correlation(X):
    # compute the total correlation C of a set of random variables X_1,...,X_n such that 
    # C(X_1,...,X_n) = H(X_1) + ... + H(X_n) - H(X_1,...,X_n)
    return drv.information_multi(X)


''' computes the Shapley Value using the characteristic function '''


class ShapleyValueCalculator:
    def __init__(self, char_function, subset_generator_function, subset_size = -1, cpu_count = 4):
    
        '''
        
        char_function =                         'entropy' or 'total_correlation'
        subset_generator_function =             sets to consider when computing the Shapley Values
        subset_size =                           bound or the subsets dimension (only relevant when subset_generator_function = 'subsets_bounded')
        
        '''
        
        self.char_function = char_function
        self.cpu_count = cpu_count
        self.subset_generator_function = subset_generator_function
        self.subset_size = subset_size

        self.value_dict = {}         
        self.data_size = 0
        self.data = 0
        
     
    
    def value_function_helper(self, subset):
        key = ""
        for my_set in subset:
            key = key + "["+ str(my_set) + "]"
        selected_data = self.data[subset, :]   
        value = eval(self.char_function)(selected_data)
        return (key,value)
        
                                                              
    def calculate_value_functions(self, data):
        self.data = data
        
        if self.subset_generator_function == "subsets":
            subset_generator = eval("sf." + self.subset_generator_function)(data)
        else:
            subset_generator = eval("sf." + self.subset_generator_function)(data, self.subset_size)
        
        pool = mp.Pool(self.cpu_count)
        
        [(data, feature_number) for feature_number in range(data.shape[0])]
        results = pool.starmap(self.value_function_helper, [(subset, ) for subset in subset_generator] )
        self.value_dict = dict(results)
                                                               
    def calculate_sv(self, data, feature_number):
        shapley_sum = 0
        normalize_sum = 0
        
        sets = self.value_dict.keys()
        feature_string = "[" + str(feature_number) + "]"
        for my_set in sets:
            if feature_string in my_set:
                if len(my_set) < 6:
                    normalize_sum += math.factorial(data.shape[0] - 1)
                    continue
                try:
                    value_S_k = self.value_dict[my_set]
                except:
                    continue
                try:
                    value_S = self.value_dict[my_set.replace(feature_string, "")]
                    l = my_set.replace(feature_string, "").count("[")
                    permutations_covered = math.factorial((data.shape[0]-l-1)) * math.factorial(l)
                    
                    normalize_sum += permutations_covered
                    
                    fac = permutations_covered
                    shapley_sum = shapley_sum + (value_S_k - value_S)*fac
                except:
                    continue
                            

        shapley_sum = shapley_sum/normalize_sum
        return shapley_sum
    
                                                  
    def calculate_SVs(self, data):
        shapley_results = []
        if not bool(self.value_dict) or self.data_size != data.shape[0]:
                                         
                 
            self.calculate_value_functions(data)
            self.data_size = data.shape[0]
        for i in range(self.data_size):
            shapley_results.append(self.calculate_sv(data, i))
        shapley_results = [i for i in shapley_results if i != 0]
        if len(shapley_results) == 0: return [0]
        return shapley_results
               
    
    
    
    
    
    

