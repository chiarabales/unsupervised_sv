import numpy as np
import shapley_calculation as sv


class FeatureSelectionCalculator:
    def __init__(self, data, SVC):
        self.data = data
        self.SVC = SVC
        
        '''
        takes as input the data and the SVC
        
        EXAMPLE:
        
        SVC = SV.ShapleyValueCalculatorFast("total_correlation", "subsets_bounded", subset_size)
        SVFR = FS.FeatureSelectionCalculator(mydata, SVC)
        S = SVFR.SVFR()
        print(S)
        
        '''
    
    def SVFS(self, epsilon, feature_count = 1000):

        feature_numbers = np.arange(self.data.shape[0])
        selected_features = [ np.argmax(self.SVC.calculate_SVs(self.data))]
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
            shapley_values = self.SVC.calculate_SVs( \
                           np.delete(data_cpy, neglected_features + selected_features, axis = 0))

            best_feature = np.argmax(shapley_values)
            selected_features.append(feature_names[best_feature])
    
        return selected_features
    
    def SVFR(self, feature_count = 1000):

        feature_numbers = np.arange(self.data.shape[0])
        selected_features = [ np.argmax(self.SVC.calculate_SVs(self.data))]
        
        for i in range(1, min(feature_count, self.data.shape[0])):
        
            if (len(selected_features) >= len(feature_numbers)):
                break
            
            remaining_data = np.delete(self.data.copy(), selected_features, axis = 0)
            feature_names = np.delete(feature_numbers.copy(), selected_features, axis = 0)    
            shapley_values = self.SVC.calculate_SVs(remaining_data)
                
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
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        