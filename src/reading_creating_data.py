# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:09:19 2020

@author: Chiara Balestra
"""

import numpy as np

import os
import pandas as pd
import seaborn as sns
import random
sns.set()

_data_path = os.path.join('..', 'data')
np.random.seed(42)

# creation of datasets, and import of real datsets

def create_data_1():
    data = np.array([[0,1,1,0],[0,1,0,1],[1,1,1,1]])
    return data

def create_data_2(size):
    x0 = np.random.randint(0, 3, size)
    x1 = np.random.randint(0, 5, size)
    x3 = np.random.randint(0, 2, size)
    x4 = np.random.randint(x3.max(), x3.max()+2, size)
    x2 = 2*x0+x1-x3
    x5 = 3*x4
    data = np.concatenate((x0, x1.T, x2.T, x3.T, x4.T, x5.T))
    data = np.reshape(data, (6 , size))
    return data

#creatre data to test the redunddancy awareness
def create_data_3(size):
    
    x0 = np.random.randint(0, 3, size)
    x1 = np.random.randint(0, 5, size)
    x2 = 2*x0
    x3 = 3*x1
    
    x4 = np.random.randint(0, 4, size)
    x5 = np.random.randint(0, 3, size)
    x6 = -x4
    x7 = 2*x5
    
    x8 = np.random.randint(0, 4, size)
    #x9 = np.random.randint(0, 2, size)
    x10 = -2* x8 
    x11 = 3*x10
    
    data = np.concatenate((x0, x2.T, x4.T, x6.T, x8.T, x10.T))
    data = np.reshape(data, (6 , size))
    return data

from sklearn.preprocessing import LabelEncoder


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
    return mydata, label

def open_big5_random(feature_number, seed = 42, samples = 10000):

    path = os.path.join(_data_path, "big5.csv")
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

def open_big5_working_cluster():
    path = os.path.join(_data_path, "big5.csv")
    mydata =  pd.read_csv(path, sep = "\t")
    mydata.drop(mydata.columns[50:], axis=1, inplace=True)
    
    mydata.drop(mydata.columns[30:], axis=1, inplace=True)
    mydata.drop(mydata.columns[24:], axis=1, inplace=True)
    mydata.drop(mydata.columns[13:20], axis=1, inplace=True)
    mydata.drop(mydata.columns[3:10], axis=1, inplace=True)

    #mydata.drop(mydata.columns[30:], axis=1, inplace=True)
    columns = mydata.columns
    mydata.dropna(inplace = True)
    mydata = mydata.head(10000)
    mydata = np.asarray(mydata)
    mydata = mydata.T
    mydata = mydata.astype('int')
    return mydata, columns

def open_big5_random_cluster(features_per_cluster, cluster_count):
    path = os.path.join(_data_path, "big5.csv")
    mydata =  pd.read_csv(path, sep = "\t")
    mydata.drop(mydata.columns[50:], axis=1, inplace=True)
    
    cluster1 = mydata.iloc[:, 0:10]
    cluster2 = mydata.iloc[:, 10:20]
    cluster3 = mydata.iloc[:, 20:30]
    cluster4 = mydata.iloc[:, 30:40]
    cluster5 = mydata.iloc[:, 40:]
    
    cluster_list = [cluster1, cluster2, cluster3, cluster4, cluster5]
    random_index = np.array(range(5))
    random.shuffle(random_index)
    selected_clusters = []
    for i in range(cluster_count):
        features = cluster_list[random_index[i]].sample(n=features_per_cluster,axis='columns')
        selected_clusters.append(features)
        
    result = pd.concat(selected_clusters, axis = 1)
    columns = result.columns

    result.dropna(inplace = True)
    result = result.head(10000)
    result = np.asarray(result)
    result = result.T
    result = result.astype('int')
    return result, columns

def open_big5_selected_cluster(features_per_cluster_list, clusters):
    path = os.path.join(_data_path, "big5.csv")
    mydata =  pd.read_csv(path, sep = "\t")
    mydata.drop(mydata.columns[50:], axis=1, inplace=True)
    
    cluster1 = mydata.iloc[:, 0:10]
    cluster2 = mydata.iloc[:, 10:20]
    cluster3 = mydata.iloc[:, 20:30]
    cluster4 = mydata.iloc[:, 30:40]
    cluster5 = mydata.iloc[:, 40:]
    
    cluster_list = [cluster1, cluster2, cluster3, cluster4, cluster5]
    selected_clusters = []
    for i in range(len(clusters)):
        features = cluster_list[clusters[i]].iloc[:, 0:features_per_cluster_list[i]]
        selected_clusters.append(features)
        
    result = pd.concat(selected_clusters, axis = 1)
    columns = result.columns

    result.dropna(inplace = True)
    result = result.head(1000)
    result = np.asarray(result)
    result = result.T
    result = result.astype('int')
    return result, columns

def open_fifa_random(feature_number, seed = 42, ninenine = False):
    path = os.path.join(_data_path, "players_20.csv")
    mydata =  pd.read_csv(path, sep = ",")
    
    #for entry in mydata.columns: 
    #    print(entry)
    
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
                "defending_marking","defending_standing_tackle","defending_sliding_tackle"]
        
    all_features_99 = ["overall",\
                "potential",\
                "pace","shooting",\
                "passing","dribbling","defending","physic","attacking_crossing",\
                "attacking_finishing","attacking_heading_accuracy",\
                "attacking_short_passing","attacking_volleys","skill_dribbling",\
                "skill_curve","skill_fk_accuracy","skill_long_passing","skill_ball_control",\
                "movement_acceleration","movement_sprint_speed","movement_agility",\
                "movement_reactions","movement_balance","power_shot_power",\
                "power_jumping","power_stamina","power_strength","power_long_shots",\
                "mentality_aggression","mentality_interceptions","mentality_positioning",\
                "mentality_vision","mentality_penalties","mentality_composure",\
                "defending_marking","defending_standing_tackle","defending_sliding_tackle"]
    
    if ninenine == True:
        all_features = all_features_99
        
    result= mydata.loc[0:2000, all_features]

    result.dropna(inplace = True)
    result = result.sample(n=feature_number,axis='columns', random_state = seed)
    columns = result.columns
    result = np.asarray(result)
    result = result.T
    result = result.astype("int")
    
    return result, columns



def open_fifa_data_working_cluster():
    path = os.path.join(_data_path, "players_20.csv")
    mydata =  pd.read_csv(path, sep = ",")
    
    mydata = mydata[mydata.player_positions != "GK"]
    
#     cluster1 = mydata.loc[0:5000,["overall"]]#, "potential", "value_eur", "wage_eur", "international_reputation", "release_clause_eur"]]
    cluster2 = mydata.loc[0:5000, ["pace", "movement_acceleration", "movement_sprint_speed",  "movement_agility", "movement_reactions", "movement_balance"]]
    cluster3 = mydata.loc[0:5000, ["attacking_crossing", "attacking_finishing", "attacking_volleys", "attacking_heading_accuracy", "attacking_short_passing" ]]
    cluster4 = mydata.loc[0:5000, ["skill_curve", "skill_fk_accuracy", "skill_long_passing","skill_dribbling", "skill_ball_control"]]
    cluster5 = mydata.loc[0:5000, ["defending", "defending_standing_tackle", "defending_sliding_tackle", "defending_marking"]]
    cluster6 = mydata.loc[0:5000, ["mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure"]]
    
    frames = [cluster2, cluster3, cluster5]
    result = pd.concat(frames, axis = 1)
    columns = result.columns
    result.dropna(inplace = True)
    result = np.asarray(result)
    result = result.T
    result = result.astype("int")
    
    return result, columns