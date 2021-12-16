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

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

brett_boris_files_repeat = sorted(glob('./boris_training_annotations/brett_reannotation/*.csv'))
donnie_boris_files_repeat = sorted(glob('./boris_training_annotations/donnie_reannotation/*.csv'))

def truncate(l):
    return l[:5]

#Make sure we only look at the first five in each
tracking_files = truncate(tracking_files)
video_files = truncate(video_files)
donnie_boris_files = truncate(donnie_boris_files)
brett_boris_files = truncate(brett_boris_files)
brett_boris_files_repeat = truncate(brett_boris_files_repeat)
donnie_boris_files_repeat = truncate(donnie_boris_files_repeat)

def preprocess_labels(dlc_files, labelsets, **kwargs):

    datasets = []
    names = []
    for k,v in labelsets.items():
        metadata = clone_metadata(dlc_files, label_files = v, **kwargs)
        datasets.append(VideosetDataFrame(metadata))
        names.append(k)

    dataset = datasets[0]

    col_names = []
    for k, d in zip(names, datasets):
        cn = f'labels_{k}'
        dataset.data[cn] = d.labels
        col_names.append(cn)

    dataset.data['labels_intersect'] = np.min(dataset.data[col_names].to_numpy(), axis = 1)    
    dataset.label_cols = 'labels_intersect'
    return dataset

#Create features, etc for training and test datasets
labelsets = {'brett': brett_boris_files, 'donnie': donnie_boris_files, 'brett_repeat': brett_boris_files_repeat, 'donnie_repeat': donnie_boris_files_repeat}
dataset = preprocess_labels(tracking_files, labelsets, video_files = video_files,
                            units = units, fps = fps, resolution = resolution, frame_length = frame_length)



#Check precision and recall bw Brett/Donnie, Brett/Brett, Donnie/Donnie
def print_scores(dataset, col1, col2):
    acc = accuracy_score(dataset.data[col1], dataset.data[col2])
    f1 = f1_score(dataset.data[col1], dataset.data[col2])
    pr = precision_score(dataset.data[col1], dataset.data[col2])
    re = recall_score(dataset.data[col1], dataset.data[col2])
    print("Acc", acc, "F1", f1, 'precision', pr, 'recall', re)

print("Consistency metrics (Brett/Donnie)")
print_scores(dataset, 'labels_brett', 'labels_donnie')

print("Consistency metrics (Brett/Brett)")
print_scores(dataset, 'labels_brett', 'labels_brett_repeat')

print("Consistency metrics (Donnie/Donnie)")
print_scores(dataset, 'labels_donnie', 'labels_donnie_repeat')
################################################################################
