#################################
## Animal pilot study pipeline ##
#################################

#TODO

# * Finish refining labels
# * Check quality of DLC tracks
# * Check DLC consistency w Brett's labels
# * Check QC of model predictions... Do a hand sample of a few videos. Also, plot the ML-predicted interaction times
#   w simple metrics, like the time spent with one mouse's head close to the other's body. 
# * Refine...models:
#   * Maybe add DLC tracking likelihood as extra features
#   * Inspect by eye the interaction times of the two groups... do I see an effect by eye?
# * Unsupervised learning? TSNE embedding of features, colored by behavior predictions
#   For this, can also try using a compressed version of the feature space (using an autoencoder first), before giving
#   to TSNE. 
#   Can also color TSNE embedding by control group
#   Can also compute embedding for simple measures of interest (like how often different body parts are from each other)
#   -- color this by control group
# * Gantt plots of predictions, by each group

#DONE
# * Compute predicted interaction times for all videos
# * Make videos of predictions

#Seperate TODO
# * A simple analysis of the types of features that help with behavior classification
#   Compare: DL + MARS, DL only, MARS only, DL + MARS + likelihood of each point

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

boris_files = sorted(glob('./boris_training_annotations/brett_initial/*.csv'))

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

def preprocess(dlc_files, **kwargs):

    metadata = clone_metadata(dlc_files, **kwargs)
    dataset = VideosetDataFrame(metadata)

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

#Create features, etc for training and test datasets
dataset = preprocess(tracking_files, label_files = boris_files, video_files = video_files,
                     units = units, fps = fps, resolution = resolution, frame_length = frame_length)

test_dataset = preprocess(test_tracking_files, video_files = test_video_files, 
                          units = units, fps = fps, resolution = resolution, frame_length = frame_length)

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
acc = accuracy_score(dataset.labels, predictions)
f1 = f1_score(dataset.labels, predictions)
pr = precision_score(dataset.labels, predictions)
re = recall_score(dataset.labels, predictions)
print("Validation metrics")
print("Acc", acc, "F1", f1, 'precision', pr, 'recall', re)
dataset.data['prediction'] = predictions

#Refit on all data
model.fit(dataset.features, dataset.labels)

#Save model here to not have to worry about training next time
save_sklearn_model(model, './analysis/xgb_model_all_training.pkl')

#Predict on test data
test_predictions = model.predict(test_dataset.features)
test_dataset.data['prediction'] = test_predictions

#Save dataframe for later analysis
dataset.save('./analysis/dataset_training.pkl')
test_dataset.save('./analysis/dataset_test.pkl')

#Make videos of performance in training data
dataset.make_movie({'label':'invest.', 'prediction':'predicted invest.'}, './analysis/videos/training/')

#Make videos of predictions in test data
test_dataset.make_movie({'prediction':'predicted invest.'}, './analysis/videos/test/')