import numpy as np
import get_feats
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

################## OPTIONS ##################

# Save Results
rdf_save_bool = 0

# Split data
split_training_data_bool = 1

# Show Confusion Matrix
show_conf_mat = 1

# RDF options
min_estimator = 1
max_estimator = 301
step_size = 10

################## VAR INIT ##################

time_now = int(time.time())
oob_error_array = []
vali_error_array = []
parent_dir = "/media/autobuntu/chonk/chonk/git_repos/py_rdf_class/Py_Decision_Trees"

##############################################

# Getting desired features
col_array = get_feats.get_feats()

# Training Data Import
train_data_all = pd.read_csv('Training_Data/RAN/RAN_BULK.csv')


# Split data - If needed, the total amount of data can be split into training/testing etc
# Otherwise, it will load dedicated validation data seperate from the training data
if split_training_data_bool:

    terrain_types = train_data_all[train_data_all.columns[-1]]

    # Splitting Training Data
    train_data_all, vali_data_all, terrain_types_train, terrain_types_vali = train_test_split(train_data_all, 
                                                                                                terrain_types, 
                                                                                                test_size = 0.20,
                                                                                                stratify=terrain_types)

    # Handling Training Data
    train_data = train_data_all[train_data_all.columns[col_array]]
    train_data = np.nan_to_num(train_data,nan= 0, posinf = 0, neginf=0)

    # Handling Validation Data
    vali_data = vali_data_all[vali_data_all.columns[col_array]]
    vali_data = np.nan_to_num(vali_data,nan= 0, posinf = 0, neginf=0)

    # print(train_data.shape)
    # print(vali_data.shape)

else:

    # Handling Training Data
    train_data = train_data_all[train_data_all.columns[col_array]]
    train_data = np.nan_to_num(train_data,nan= 0, posinf = 0, neginf=0)
    terrain_types_train = train_data_all[train_data_all.columns[-1]]
    
    # Validation File Import
    vali_data_all = pd.read_csv('Validation_Data/RANSAC_VALIDATION.csv')

    # Handling Validation Data
    vali_data = vali_data_all[vali_data_all.columns[col_array]]
    vali_data = np.nan_to_num(vali_data,nan= 0, posinf = 0, neginf=0)
    print(vali_data.shape)
    terrain_types_vali = vali_data_all[vali_data_all.columns[-1]]

# Creating Tune Parameter List
# parameters = {'min_samples_split':list(range(1,50,1)),
# 'min_samples_leaf':list(range(1,50,1)),
# 'max_features':np.linspace(0.5,0.95,5),
# 'ccp_alpha':np.linspace(0.0,0.01,10),
# 'n_estimators':list(range(5,255,10))}

# pipe = Pipeline([
#     ('rf', RandomForestClassifier())
# ])

parameters = {
    'n_estimators':list(range(5,255,10)),
    'max_depth':list(range(5,105,10)),
    'min_samples_split':[2,3],
    'min_samples_leaf':[3,5]}

# print(parameters)

# Creating the RDF parameters
# Parameters are explained here - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Principal differences from standard is the use of all cores (n_jobs = -1), verbose=1,  and warm_start=True.
rdf_clf = RandomForestClassifier()#n_estimators=1,  
# criterion='gini', 
# max_depth=5, 
# min_samples_split=10, 
# min_samples_leaf=3, 
# min_weight_fraction_leaf=0.0, 
# max_features=0.5, 
# max_leaf_nodes=None, 
# min_impurity_decrease=0.0, 
# bootstrap=True, 
# oob_score=True, 
# n_jobs=-1, 
# random_state=None, 
# verbose=1, 
# warm_start=True, 
# class_weight=None,
# ccp_alpha=0.0, 
# max_samples=0.9)

# Testing the RDF

grid = GridSearchCV(rdf_clf, param_grid=parameters)

grid.fit(train_data, terrain_types_train)

print(grid.score)