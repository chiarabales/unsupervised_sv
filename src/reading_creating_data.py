import numpy as np

import os
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder


_data_path = os.path.join('..', 'data')


def create_data(size):
    
    random.seed(46)

    x0 = np.random.randint(0, 3, size)
    x1 = np.random.randint(0, 5, size)
    x2 = 2*x0
    x3 = x1 - x0
    
    x4 = np.random.randint(0, 4, size)
    x5 = np.random.randint(0, 3, size)
    x6 = -x4
    x7 = 2*x5 - x4
    
    x8 = np.random.randint(0, 8, size)
    x9 = np.random.randint(0, 7, size)
    x10 = -2* x8 
    x11 = 3*x10 + x8
    
    data = np.concatenate((x0, x1.T, x2.T, x3.T, x4.T, x5.T, x6.T, x7.T, x8.T, x9.T, x10.T, x11.T))
    data = np.reshape(data, (12 , size))
    return data.T

# ____________________________________________________________________________________________________


def open_breast_cancer():
    path = os.path.join(_data_path, "breast-cancer.txt")
    data = pd.read_csv(path, sep = ",", header = None, names = ['class', 'age', 'menopause', 'tumor_size', 'inv_nodes', 'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat'])
    data['age'] = data['age'].replace(['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    data['menopause'] = data['menopause'].replace(['lt40', 'ge40', 'premeno'], [1, 2, 3])
    data['tumor_size'] = data['tumor_size'].replace(['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    data['inv_nodes'] = data['inv_nodes'].replace(['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    data['node_caps'] = data['node_caps'].replace(['yes', 'no', '?'], [1, 0, 99])
    data['breast'] = data['breast'].replace(['left', 'right'], [1, 2])
    data['breast_quad'] = data['breast_quad'].replace(['left_up', 'left_low', 'right_up', 'right_low', 'central', '?'], [11, 10, 21, 20, 0, 99])
    data['irradiat'] = data['irradiat'].replace(['yes', 'no'], [1, 0])
    data.dropna(inplace = True)
    data2 = np.asarray(data)
    mydata = data2
    Y = np.delete(mydata, 0, axis = 1)
    Y = Y.astype('int')
    label = mydata[:,0]
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')
    label = label.astype('uint8')
    mydata = Y.astype('int')
    return mydata.T, label

# ____________________________________________________________________________________________________


def open_big5_random(feature_number, seed = 42, samples = 10000):

    path = os.path.join(_data_path, "data-final.csv")
    mydata =  pd.read_csv(path, sep = "\t")
    mydata.drop(mydata.columns[50:], axis=1, inplace=True)
    
    mydata = mydata.sample(n=feature_number,axis='columns', random_state = seed)
    
    mydata.dropna(inplace = True)
    mydata = mydata.head(samples)
    columns = mydata.columns
    mydata = np.asarray(mydata)
    mydata = mydata.T
    mydata = mydata.astype('int')
    return mydata, columns

# ____________________________________________________________________________________________________


def open_fifa_random(feature_number, seed = 42, samples = 5000):
    
    path = os.path.join(_data_path, "players_20.csv")
    mydata =  pd.read_csv(path, sep = ",")
    
    mydata = mydata[mydata.player_positions != "GK"]
    
    all_features = ["age","height_cm","weight_kg","overall",\
                "potential","value_eur","wage_eur","international_reputation","weak_foot",\
                "skill_moves","release_clause_eur","team_jersey_number","pace","shooting",\
                "passing","dribbling","defending","physic","attacking_crossing",\
                "attacking_finishing","attacking_heading_accuracy",\
                "attacking_short_passing","attacking_volleys","skill_dribbling",\
                "skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control",\
                "movement_acceleration","movement_sprint_speed","movement_agility",\
                "movement_reactions","movement_balance","power_shot_power",\
                "power_jumping","power_stamina","power_strength","power_long_shots",\
                "mentality_aggression","mentality_interceptions","mentality_positioning",\
                "mentality_vision","mentality_penalties","mentality_composure",\
                "defending_standing_tackle","defending_sliding_tackle"]
        
    result= mydata.loc[0:samples, all_features]

    result.dropna(inplace = True)
    result = result.sample(n=feature_number,axis='columns', random_state = seed)
    columns = result.columns
    result = np.asarray(result)
    result = result.T
    result = result.astype("int")
    
    return result, columns



