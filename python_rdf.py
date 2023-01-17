import csv
import numpy as np
import get_feats
import sys
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import joblib

################## OPTIONS ##################

# Save Results
rdf_save_bool = 1

# RDF options
min_estimator = 5
max_estimator = 305
step_size = 5

################## VAR INIT ##################

time_now = int(time.time())
oob_error_array = []
vali_error_array = []
parent_dir = "/media/autobuntu/chonk/chonk/git_repos/py_rdf_class/Py_Decision_Trees"

##############################################

# Getting desired features
col_array = get_feats.get_feats()

# Training Data Import
td_df = pd.read_csv('Training_Data/RANSAC_TRAINING.csv')
td_df_to_rdf = td_df[td_df.columns[col_array]]
td_df_to_rdf = np.nan_to_num(td_df_to_rdf,nan= 0, posinf = 0, neginf=0)
terrain_types_train = td_df[td_df.columns[-1]]

# Validation File Import
vali_data_raw = pd.read_csv('Validation_Data/RANSAC_VALIDATION.csv')
vali_data = vali_data_raw[vali_data_raw.columns[col_array]]
vali_data = np.nan_to_num(vali_data,nan= 0, posinf = 0, neginf=0)
terrain_types_vali = vali_data_raw[vali_data_raw.columns[-1]]

# Creating the RDF parameters
# Parameters are explained here - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Principal differences from standard is the use of all cores (n_jobs = -1), verbose=1, max_depth=10,100,200, and warm_start=True.
rdf_clf = RandomForestClassifier(n_estimators=100,  
criterion='gini', 
max_depth=35, 
min_samples_split=2, 
min_samples_leaf=1, 
min_weight_fraction_leaf=0.0, 
max_features='sqrt', 
max_leaf_nodes=20, 
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

# Save locations
if rdf_save_bool:
    rdf_export_dir = 'RDF_RAN_' + str(rdf_clf.max_depth) + str(time_now)
    rdf_export_path = os.path.join(parent_dir, rdf_export_dir)
    os.mkdir(rdf_export_path)


# Training, Testing, and Validation
for i in range(min_estimator, max_estimator+1, step_size):

    # Training
    rdf_clf.set_params(n_estimators = i)
    rdf_clf.fit(td_df_to_rdf, terrain_types_train)

    # Testing
    oob_error = 1 - rdf_clf.oob_score_
    oob_error_array.append((i,oob_error))

    # Validation
    y_pred=rdf_clf.predict(vali_data)
    vali_error_array.append((i,metrics.accuracy_score(terrain_types_vali, y_pred)))

    # Saving the Trees
    if rdf_save_bool:
        rdf_pickle_name = "RDF_" + str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".joblib"
        joblib.dump(rdf_clf, os.path.join(rdf_export_path,rdf_pickle_name))

# Saving the validation results
if rdf_save_bool:
    vali_pickle_name = str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".joblib"
    joblib.dump(vali_error_array, os.path.join(rdf_export_path,vali_pickle_name))

# Grabbing results from train test validate
oob_x_coords = [coord[0] for coord in oob_error_array]
oob_y_coords = [coord[1] for coord in oob_error_array]
vali_x_coords = [coord[0] for coord in vali_error_array]
vali_y_coords = [coord[1] for coord in vali_error_array]

# Plotting
fig = plt.figure(figsize=(15,15))
plt.plot(oob_x_coords, oob_y_coords)
plt.plot(vali_x_coords, vali_y_coords)
plt.xlabel('n_estimators')
plt.ylabel('Error %')
plt.show()

# Save Plot
if rdf_save_bool:
    plot_name = str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".png"
    fig.savefig(os.path.join(rdf_export_path,plot_name), bbox_inches='tight', dpi=500)
