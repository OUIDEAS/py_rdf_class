import csv
import numpy as np
import get_feats
import sys
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.metrics import confusion_matrix

################## OPTIONS ##################

# Save Results
rdf_save_bool = 0

# Split data
split_training_data_bool = 0

# Show Confusion Matrix
show_conf_mat = 1

# RDF options
min_estimator = 1
max_estimator = 201
step_size = 10

# Plot style
plt.style.use('dark_background')

################## VAR INIT ##################

time_now = int(time.time())
oob_error_array = []
vali_error_array = []
# parent_dir = "/media/autobuntu/chonk/chonk/git_repos/py_rdf_class/Py_Decision_Trees"
parent_dir = "/home/rhett/git_repos/python_rdf/Py_Decision_Trees"
old_diag_score = 0
conf_mat_store = np.zeros((4,4))

##############################################

# Getting desired features
col_array = get_feats.get_feats()

# Training Data Import
# train_data_all = pd.read_csv('Training_Data/RAN/RAN_BULK_ALL_&_MANUAL.csv')
# train_data_all = pd.read_csv('Training_Data/RAN/Train_All_Feats_data.csv')
train_data_all = pd.read_csv('Validation_Data/RAN/Test_All_Feats_data.csv')

# Split data - If needed, the total amount of data can be split into training/testing etc
# Otherwise, it will load dedicated validation data seperate from the training data
if split_training_data_bool:

    terrain_types = train_data_all[train_data_all.columns[-1]]

    # Splitting Training Data
    train_data_all, vali_data_all, terrain_types_train, terrain_types_vali = train_test_split(train_data_all, 
                                                                                                terrain_types, 
                                                                                                test_size = 0.50,
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
    train_data = train_data_all[train_data_all.columns[:-1]]
    train_data = np.nan_to_num(train_data,nan= 0, posinf = 0, neginf=0)
    terrain_types_train = train_data_all[train_data_all.columns[-1]]
    
    # Validation File Import
    # vali_data_all = pd.read_csv('Validation_Data/RAN/Test_All_Feats_data.csv')
    vali_data_all = pd.read_csv('Training_Data/RAN/Train_All_Feats_data.csv')

    # Handling Validation Data
    vali_data = vali_data_all[vali_data_all.columns[:-1]]
    vali_data = np.nan_to_num(vali_data,nan= 0, posinf = 0, neginf=0)
    terrain_types_vali = vali_data_all[vali_data_all.columns[-1]]

# Creating the RDF parameters
# Parameters are explained here - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# Principal differences from standard is the use of all cores (n_jobs = -1), verbose=1,  and warm_start=True.
rdf_clf = RandomForestClassifier(n_estimators=1,  
criterion='gini', 
max_depth=10, 
min_samples_split=2, 
min_samples_leaf=3, 
min_weight_fraction_leaf=0.40, 
max_features='sqrt', 
max_leaf_nodes=3, 
min_impurity_decrease=0.0, 
bootstrap=True, 
oob_score=True, 
n_jobs=-1, 
random_state=None, 
verbose=1, 
warm_start=True, 
class_weight=None,
ccp_alpha=0.0, 
max_samples=0.50)

# Save locations
if rdf_save_bool:

    rdf_export_dir = 'RDF_RAN_' + str(rdf_clf.max_depth) + str(time_now)
    rdf_export_path = os.path.join(parent_dir, rdf_export_dir)
    os.mkdir(rdf_export_path)


# Training, Testing, and Validation
for n_est_idx in range(min_estimator, max_estimator+1, step_size):

    # Training
    rdf_clf.set_params(n_estimators = n_est_idx)
    rdf_clf.fit(train_data, terrain_types_train)

    # Testing
    oob_error = 1 - rdf_clf.oob_score_
    oob_error_array.append((n_est_idx,oob_error))

    # Validation
    y_pred=rdf_clf.predict(vali_data)
    vali_error_array.append((n_est_idx,(1-metrics.accuracy_score(terrain_types_vali, y_pred)))) # ACCURACY NOT ERROR LOL

    # Saving the Trees
    if rdf_save_bool:
        rdf_pickle_name = "RDF_" + str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".joblib"
        joblib.dump(rdf_clf, os.path.join(rdf_export_path,rdf_pickle_name))

    if show_conf_mat:
        conf_mat = confusion_matrix(terrain_types_vali, y_pred)
        # print(type(conf_mat))
        new_diag_score = sum(conf_mat.diagonal())
        if new_diag_score > old_diag_score:
            old_diag_score = new_diag_score
            conf_mat_store = conf_mat
            num_trees = n_est_idx



# Saving the validation results
if rdf_save_bool:
    vali_pickle_name = str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".joblib"
    joblib.dump(vali_error_array, os.path.join(rdf_export_path,vali_pickle_name))

# Grabbing results from train test validate
oob_x_coords = [coord[0] for coord in oob_error_array]
oob_y_coords = [coord[1] for coord in oob_error_array]
vali_x_coords = [coord[0] for coord in vali_error_array]
vali_y_coords = [coord[1] for coord in vali_error_array]

# Printing best results
print(old_diag_score)
print(conf_mat_store)
print(num_trees)

# Plotting
fig = plt.figure(figsize=(15,15))
plt.plot(oob_x_coords, oob_y_coords, 'o-', label = 'OOB')
plt.plot(vali_x_coords, vali_y_coords, 'o-', label = 'VALI')
plt.xlabel('n_estimators')
plt.ylabel('Error %')
plt.legend()
plt.show()

# Save Plot
if rdf_save_bool:
    plot_name = str(rdf_clf.n_estimators) + "_" + str(rdf_clf.max_depth) + ".png"
    fig.savefig(os.path.join(rdf_export_path,plot_name), bbox_inches='tight', dpi=500)
