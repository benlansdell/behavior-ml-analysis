#################################
## Animal pilot study pipeline ##
#################################

# Here we:
# * load in both Brett's and Donnie's BORIS annotations of the same set of videos
# * Compute a new set of behavior labels that is the AND of the two sets (i.e. only label behaviors that both agree on)
# * Check that we don't lose too much data by doing this... a loss of upto 20% may be ok?
# * Retrain the ML model
# * Compute the CV performance for this new set

from glob import glob 
from behaveml import VideosetDataFrame, clone_metadata
from behaveml import mars_feature_maker, cnn_probability_feature_maker, interpolate_lowconf_points, \
                     save_sklearn_model

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import os 
import numpy as np

#Video configs
frame_length = None              # (float) length of entire horizontal shot
units = None                     # (str) units frame_length is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels

#Our training data
tracking_files = sorted(glob('./dlc_training_tracks/*improved.csv'))
video_files = sorted(glob('./dlc_training_videos/*.avi'))

donnie_boris_files = sorted(glob('./boris_training_annotations/donnie_initial/*.csv'))
brett_boris_files = sorted(glob('./boris_training_annotations/brett_initial/*.csv'))

#Our test data
test_tracking_files = sorted(glob('./all_videos/*DLC_dlcrnetms5_pilot_studySep24shuffle1_100000_el_filtered.csv'))
test_video_files = sorted(glob('./all_videos/*.avi'))

#Make sure their order matches....
# #Map videos to BORIS annotations
# e3v813a-20210610T120637-121213_reencode.avi -> DLC1
# e3v813a-20210610T121558-122141_reencode.avi -> DLC2
# e3v813a-20210610T122332-122642_reencode.avi -> DLC3
# e3v813a-20210610T122758-123309_reencode.avi -> DLC4
# e3v813a-20210610T123521-124106_reencode.avi -> DLC5
# e3v813a-20210714T091732-092358.avi -> DLC_Set2_1
# e3v813a-20210714T092722-093422.avi -> DLC_Set2_2
# e3v813a-20210714T094126-094745.avi -> DLC_Set2_3
# e3v813a-20210714T095234-095803.avi -> DLC_Set2_4
# e3v813a-20210714T095950-100613.avi -> DLC_Set2_5

def preprocess(dataset):
    print("Interpolating low-confidence tracking points")
    interpolate_lowconf_points(dataset)

    print("Calculating MARS features")
    dataset.add_features(mars_feature_maker, 
                         featureset_name = 'MARS', 
                         add_to_features = True)

    #Note: by default this keras code will try to use CUDA. 
    print("Calculating 1D CNN pretrained network features")
    dataset.add_features(cnn_probability_feature_maker, 
                         featureset_name = '1dcnn', 
                         add_to_features = True)
    return dataset

def preprocess_labels(dlc_files, labelsets, **kwargs):

    datasets = []
    names = []
    for k,v in labelsets.items():
        metadata = clone_metadata(dlc_files, label_files = v, **kwargs)
        datasets.append(VideosetDataFrame(metadata))
        names.append(k)

    dataset = preprocess(datasets[0])

    col_names = []
    for k, d in zip(names, datasets):
        cn = f'labels_{k}'
        dataset.data[cn] = d.labels
        col_names.append(cn)

    dataset.data['labels_intersect'] = np.min(dataset.data[col_names].to_numpy(), axis = 1)    
    dataset.label_cols = 'labels_intersect'
    return dataset

#Create features, etc for training and test datasets
labelsets = {'brett': brett_boris_files, 'donnie': donnie_boris_files}
dataset = preprocess_labels(tracking_files, labelsets, video_files = video_files,
                            units = units, fps = fps, resolution = resolution, frame_length = frame_length)

test_metadata = clone_metadata(test_tracking_files, video_files = test_video_files, 
                               units = units, fps = fps, resolution = resolution, frame_length = frame_length)
test_dataset = VideosetDataFrame(test_metadata)
test_dataset = preprocess(test_dataset)

################################################################################

#Now look at:
print("Donnie's # of behavior frames:", np.sum(dataset.data.labels_donnie))
print("Brett's # of behavior frames:", np.sum(dataset.data.labels_brett))
print("Intersection # of behavior frames:", np.sum(dataset.data.labels_intersect))
print("-- as a percentage:", 100*np.sum(dataset.data.labels_intersect)/np.sum(dataset.data.labels_donnie))

# print("Intersection # of behavior frames:", np.sum(dataset.data.label_intersect))
# print("-- as a percentage:", 100*np.sum(dataset.data.label_intersect)/np.sum(dataset.data.labels_donnie))

################################################################################

#Then go through the same training process to train the behavior model
splitter = GroupKFold(n_splits = dataset.n_videos)

#model = RandomForestClassifier()
model = XGBClassifier()

#Check the validation performance is ok before applying to test data
print("Fitting ML model with (group) LOO CV")
predictions = cross_val_predict(model, 
                                dataset.features, 
                                dataset.labels, 
                                groups = dataset.group, 
                                cv = splitter)
dataset.data['prediction'] = predictions
acc = accuracy_score(dataset.labels, dataset.data['prediction'])
f1 = f1_score(dataset.labels, dataset.data['prediction'])
pr = precision_score(dataset.labels, dataset.data['prediction'])
re = recall_score(dataset.labels, dataset.data['prediction'])
print("Validation metrics")
print("Acc", acc, "F1", f1, 'precision', pr, 'recall', re)

#Refit on all data
model.fit(dataset.features, dataset.labels)

#Save model here to not have to worry about training next time
save_sklearn_model(model, './analysis/xgb_model_all_training_brett_donnie_agree.pkl')

#Predict on test data
test_predictions = model.predict(test_dataset.features)
test_dataset.data['prediction'] = test_predictions

#Save dataframe for later analysis
dataset.save('./analysis/dataset_training_brett_donnie_agree.pkl')
test_dataset.save('./analysis/dataset_test_brett_donnie_agree.pkl')

#Make videos of performance in training data
#dataset.make_movie(['labels_intersect', 'prediction', 'labels_brett', 'labels_donnie'], './analysis/videos/training_brett_donnie_agree/')
dataset.make_movie({'prediction':'predicted invest.', 'labels_brett':'invest. (Brett)', 'labels_donnie':'invest. (Donnie)'}, './analysis/videos/training_brett_donnie_agree/')

#Make videos of predictions in test data
test_dataset.make_movie(['prediction'], './analysis/videos/test_brett_donnie_agree/')