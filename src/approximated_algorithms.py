import numpy as np 
import pandas as pd
import itertools
import random
import math

import shapley_calculation as sv
import feature_selection as fs



def build_permutations(X):
    arr = list(range(X.shape[0]))
    for value in itertools.permutations(arr, n_index + 1):
        yield list(value)


def marginal_contribution(mydata, features):
    
    ''' 
    
    computes the marginal contribution given one subset of features 
    
    '''
    element = np.random.permutation(features)
    shapleys = [0 for i in range(features)]
    for i in range(features):
        result = np.where(np.asarray(element) == i)[0][0]

        if result != 0:
            subset = list(element[:result])
            subset_with = list(element[:result+1])
            shapleys[i] += sv.total_correlation(np.asarray(mydata.loc[subset_with])) \
                                        - sv.total_correlation(np.asarray(mydata.loc[subset]))
    return shapleys


def approximated_shapley(data, approx = 'max'):
    if approx == 'max':
        approx = math.factorial(np.shape(data)[0])
    if approx > math.factorial(np.shape(data)[0]):
        approx = math.factorial(np.shape(data)[0])
        
    features = np.shape(data)[0]
    mydata = pd.DataFrame(data)
    shapleys = []

    for j in range(approx):
        shapleys.append(marginal_contribution(mydata, features))
    shapleys = np.sum(shapleys, axis = 0)
        
    shapleys[:] = [x / approx for x in shapleys]
    return shapleys




class FeatureSelectionCalculator:
    def __init__(self, data, approx = 'max'):
        self.data = data
        self.approx = approx

    
    def SVFS(self, epsilon, feature_count = 1000):
        feature_numbers = np.arange(self.data.shape[0])

        shapley_values = approximated_shapley(self.data, self.approx)
        selected_features = [np.argmax(shapley_values)]
        neglected_features = []
        
        for i in range(1, min(feature_count, self.data.shape[0])):
            unselected = [feature for feature in feature_numbers if feature not in selected_features]
            unselected = [feature for feature in unselected if feature not in neglected_features]
            feature_names = np.delete(feature_numbers.copy(), neglected_features + selected_features, axis = 0)
            corr_feat = selected_features
            correlation_values = []
            for test_feature in unselected:
                test_value = sv.entropy(self.data[[test_feature], :])\
                    + sv.entropy(self.data[corr_feat ,:])\
                    - sv.entropy(self.data[corr_feat + [test_feature], :])
                correlation_values.append(test_value)
                if  test_value > epsilon:
                    neglected_features.append(test_feature)

            if (len(neglected_features) + len(selected_features) >= len(feature_numbers)):
                break
            
            data_cpy = self.data.copy()
            feature_names = np.delete(feature_numbers.copy(), neglected_features + selected_features, axis = 0)  
            shapley_values = approximated_shapley( \
                           np.delete(data_cpy, neglected_features + selected_features, axis = 0), self.approx)
          
            best_feature = np.argmax(shapley_values)
            selected_features.append(feature_names[best_feature])
    
        return selected_features
    
    def SVFR(self, feature_count = 1000):
        feature_numbers = np.arange(self.data.shape[0])
        selected_features = [np.argmax(approximated_shapley(self.data, self.approx))]
        for i in range(1, min(feature_count, self.data.shape[0])):
            if (len(selected_features) >= len(feature_numbers)):
                break
            
            remaining_data = np.delete(self.data.copy(), selected_features, axis = 0)
            feature_names = np.delete(feature_numbers.copy(), selected_features, axis = 0)    
            shapley_values = approximated_shapley(remaining_data, self.approx)
                
            correlation_values = []
            for test_feature in range(len(remaining_data)):
                test_value = sv.entropy(remaining_data[[test_feature], :])\
                    + sv.entropy(self.data[selected_features ,:])\
                    - sv.entropy(np.concatenate((remaining_data[[test_feature], :], self.data[selected_features ,:])))
                correlation_values.append(test_value)


            shapley_values = np.array(shapley_values) - np.array(correlation_values)

            best_feature = np.argmax(shapley_values)
            selected_features.append(feature_names[best_feature])
            
        return selected_features