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
dim_x = 5
dim_z = 2


P0 = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1])
R = np.eye(dim_z) * np.array([0.1, 0.1])
Q = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1])

F = np.eye(dim_x)
H = np.array([
    [ 1, 0, 0, 0],
    [ 0, 1, 0, 0],
])

dt = 200 ** -1

tracker = KalmanTracker(P0, F, H, Q, R, show_predictions=True, max_strikes=50)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']


# Load the data to classify
data = pandas.read_pickle(os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/masked_whisker_ends'))
#data = data[(data.frame > 0) & (data.frame < 12000)].copy()


oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus
data.loc[data.tip_y < oof_y_thresh, 'length'] += oof_y_bonus

data_filtered = data[data.pixlen > 20].groupby('frame')




first_frame = sorted(data_filtered.groups.keys())[0]
for frame, observations in data_filtered:

    if frame == first_frame:
        continue
    if frame % 100 == 0:
        print frame
    

    observation_dicts = []
    observations = observations.sort_values('length', ascending=False)
    # observations = observations.sort_values('fol_y', ascending=True)
    indices = observations.index.values
    # observations['rank'] = observations['fol_y'].rank(ascending=False)

    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly = observation.length, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y

        z = np.array([tipx, tipy])

        x0 = np.array(
            [tipx, tipy, folx, foly, pixlen]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : ()
        })


    labels = tracker.detect2(observation_dicts)
    data.loc[indices, 'color_group'] = labels


# Now pickle mwe and run validate_classification_results on it
data.to_pickle(output_filename)


