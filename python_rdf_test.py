import csv
import numpy as np
import get_feats
import sys
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# td_df = training data (panda) data frame
td_df = pd.read_csv('Training_Data/RANSAC_TRAINING.csv')
print(np.shape(td_df))

# Getting desired features
col_array = get_feats.get_feats()

# Getting desired features
td_df_to_rdf = td_df[td_df.columns[col_array]]
print(np.shape(td_df_to_rdf))

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
n_jobs=None, 
random_state=None, 
verbose=1, 
warm_start=False, 
class_weight=None, 
ccp_alpha=0.0, 
max_samples=None)

# Training the RDF
rdf_clf.fit(td_df_to_rdf, terrain_types)




