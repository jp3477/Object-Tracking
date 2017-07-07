# Run the kalman filter algorithm on whisker data
import argparse
import os
import pandas
import matplotlib.pyplot as plt
import whiskvid
import numpy as np
import tables
import my
import numpy as np
from kalman import *
from filters import KalmanTracker


## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("output", 
    help='Name of the output file containing labeled whiskers')
args = parser.parse_args()
output_filename = args.output



#State is [tipx, tipy, pixlen]
#Measurement is [tipx, tipy]
dim_x = 3
dim_z = 2


P0 = np.eye(dim_x) * np.array([0.1, 0.1, 0.1])
R = np.eye(dim_z) * np.array([0.1, 0.1])
Q = np.eye(dim_x) * np.array([5, 5, 25])

F = np.eye(dim_x)
H = np.array([
    [ 1, 0, 0],
    [ 0, 1, 0],
])

dt = 200 ** -1


tracker = KalmanTracker(P0, F, H, Q, R, show_predictions=False)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']

# Load the data to classify
data = pandas.read_pickle(os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/masked_whisker_ends'))
data = data[(data.frame > 0) & (data.frame < 12000)].copy()

data['ordinal'] = 0

oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus

data_filtered = data[data.pixlen > 20].groupby('frame')

frequency_table = {}


first_frame = data_filtered.groups.keys()[0]
for frame, observations in data_filtered:

    if frame == first_frame:
        continue
    if frame % 100 == 0:
        print frame
    

    observation_dicts = []
    observations = observations.sort_values('fol_y', ascending=True)
    indices = observations.index.values
    # observations['rank'] = observations['fol_y'].rank(ascending=False)
    observation_count = len(observations)
    if not observation_count in frequency_table:
        frequency_table[observation_count] = {}
        for i in range(1, 9):
            frequency_table[observation_count][i] = []

    tracker.rankings = frequency_table
    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly, angle = observation.pixlen, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y, observation.angle
        
        angle *= np.pi / 180
        z = np.array([tipx, tipy])
        omega = ((diffs[frame]) / dt)
        # print rank / len(observations)
        x0 = np.array(
            [tipx, tipy, pixlen]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : ()
        })


    labels = tracker.detect(observation_dicts)
    data.loc[indices, 'color_group'] = labels



    for i in range(observation_count):
        label = labels[i]
        frequency_table[observation_count][label].append(i)


data['color_group'] = data['ordinal']
# Now pickle mwe and run validate_classification_results on it
data.to_pickle(output_filename)


