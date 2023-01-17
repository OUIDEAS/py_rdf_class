import csv
import numpy as np
import get_feats
import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


# td_df = training data (panda) data frame
td_df = pd.read_csv('Training_Data/RANSAC_TRAINING.csv')
# print(np.shape(td_df))

# Getting desired features
col_array = get_feats.get_feats()

# Getting desired features
td_df_to_rdf = td_df[td_df.columns[col_array]]
# print(np.shape(td_df_to_rdf))

# Eliminating the handful of random infs/nans
td_df_to_rdf = np.nan_to_num(td_df_to_rdf,nan= 0, posinf = 0, neginf=0)

# Terrain Types
terrain_types = td_df[td_df.columns[-1]]
# print(terrain_types)

# Creating the RDF parameters
rdf_clf = RandomForestClassifier(n_estimators=100,  
criterion='gini', 
max_depth=None, 
min_samples_split=100, 
min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, 
max_features='sqrt', 
max_leaf_nodes=None, 
min_impurity_decrease=0.0, 
bootstrap=True, 
oob_score=True, 
n_jobs=-1, 
random_state=None, 
verbose=1, 
warm_start=True, 
class_weight=None, 
ccp_alpha=0.0, 
max_samples=None)

# Training the RDF
# rdf_clf.fit(td_df_to_rdf, terrain_types)

# Getting OOB error - NEEDS WORK

min_estimator = 1
max_estimator = 300
step_size = 5
oob_error_array = []

for i in range(min_estimator, max_estimator+1, step_size):
    rdf_clf.set_params(n_estimators = i)
    rdf_clf.fit(td_df_to_rdf, terrain_types)
    oob_error = 1 - rdf_clf.oob_score_
    oob_error_array.append((i,oob_error))

# Does not calculate OOB per tree generation autoamtically - need to make it seperately
# error_rate = []
# for i in range(1, 100 + 1, 5):


# # print(error_rate)
# error_rate = [list(item) for item in error_rate]

x_coords = [coord[0] for coord in oob_error_array]
y_coords = [coord[1] for coord in oob_error_array]

plt.plot(x_coords, y_coords)
plt.show()

# print(oob_error)

##########################################################

# vali_data_raw = pd.read_csv('Training_Data/RANSAC_VERIFY.csv')
# vali_data = vali_data_raw[vali_data_raw.columns[col_array]]
# vali_data = np.nan_to_num(vali_data,nan= 0, posinf = 0, neginf=0)
# terrain_types_vali = vali_data_raw[vali_data_raw.columns[-1]]

# y_pred=rdf_clf.predict(vali_data)

# print("Accuracy:",metrics.accuracy_score(terrain_types_vali, y_pred))
