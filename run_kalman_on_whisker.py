# Run the kalman filter algorithm on whisker data
import argparse
import os
import sys
import pandas
import matplotlib.pyplot as plt
import whiskvid
import numpy as np
import tables
import my
import numpy as np
from kalman import *
from filters import KalmanTracker
import progressbar


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
dim_x = 6
dim_z = 2


P0 = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
R = np.eye(dim_z) * np.array([0.1, 0.1])
Q = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

F = np.eye(dim_x)
H = np.array([
    [ 1, 0, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 0, 0],
])

dt = 200 ** -1

tracker = KalmanTracker(P0, F, H, Q, R, show_predictions=False, max_object_count=12, max_strikes=50)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange', 'navy', 'turquoise', 'silver', 'yellowgreen']


# Load the data to classify
data = pandas.read_pickle(os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/masked_whisker_ends'))
data = data[(data.frame >= 0) & (data.frame < 12000)].copy()
data['color_group'] = 0
# curated_filename = os.path.expanduser(
#     '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/curated'
#     )
# curated_df = pandas.read_pickle(curated_filename)

print data


oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus
data.loc[data.tip_y < oof_y_thresh, 'length'] += oof_y_bonus

# data_filtered = data[(data.pixlen > 20) & (data.frame.isin(curated_df.frame))].groupby('frame')
data_filtered = data[data.pixlen > 20]




first_frame = sorted(data_filtered.groups.keys())[0]
bar = progressbar.ProgressBar()
current_frame = first_frame
try:
    for frame, observations in bar(data_filtered):
        current_frame = frame


        observation_dicts = []
        observations = observations.sort_values('length', ascending=False)
        # observations = observations.sort_values('fol_y', ascending=True)
        indices = observations.index.values
        observations['rank'] = observations['fol_y'].rank(ascending=False)

        for j, observation in observations.iterrows():
            pixlen, tipx, tipy, folx, foly = observation.length, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y
            rank = observation['rank']
            z = np.array([tipx, tipy])

            x0 = np.array(
                [tipx, tipy, folx, foly, pixlen, rank]
            )

            observation_dicts.append({
                "x" : x0,
                "z" : z,
                "fx_args" : ()
            })


        labels = tracker.detect2(observation_dicts)
        data.loc[indices, 'color_group'] = labels
except KeyboardInterrupt:
    data = data[data.frame < current_frame]
    data.to_pickle(output_filename)
    sys.exit()


# Now pickle mwe and run validate_classification_results on it
data.to_pickle(output_filename)


