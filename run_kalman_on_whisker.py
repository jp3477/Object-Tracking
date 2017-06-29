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
from filters import UnscentedKalmanTracker


## Parse arguments
parser = argparse.ArgumentParser(description="""
    Run the kalman filter on an example whiskers dataset.
""")
parser.add_argument("output", 
    help='Name of the output file containing labeled whiskers')
args = parser.parse_args()
output_filename = args.output



def state_function(state, dt, angle, new_folx, new_foly):
    tipx, tipy, folx, foly, pixlen, omega, rank = state[0], state[1], state[2], state[3], state[4], state[5], state[6]
    theta = -1 * np.arctan((tipy - foly) / (tipx - folx))

    # omega = (angle - theta) / dt
    v = omega * pixlen


    tipx = tipx + v * np.sin(theta) * dt
    tipy = tipy + v * np.cos(theta) * dt

    return np.array([tipx, tipy, folx, foly, pixlen, omega, rank])

def measurement_function(state):
    tipx, tipy, folx, foly, pixlen, rank = state[0], state[1], state[2], state[3], state[4], state[5]

    return np.array([tipx, tipy, folx, foly, pixlen, rank])



#State is [tipx, tipy, folx, foly, pixlen, omega]
dim_x = 7
dim_z = 6


P0 = np.eye(dim_x) * np.array([1, 1, 1, 0.5, 0.0001, 0.1, 10 ])
# R = np.eye(dim_z) * np.array([500, 500, 0.001, 0.001, 500,])
R = np.eye(dim_z) * np.array([70, 70, 100, 100, 10, 0.11])
Q = np.eye(dim_x) * np.array([11, 11, 9, 1.8 , 44, 20, 0.11])

dt = 200 ** -1

tracker = UnscentedKalmanTracker(P0, Q, R, state_function, measurement_function, dt, show_predictions=False)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}


# Load the data to classify
data = pandas.read_pickle(os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/masked_whisker_ends'))
data = data[data.frame < 2000].copy()

data['ordinal'] = 0

oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus

data_filtered = data[data.pixlen > 20].groupby('frame')
angles = data_filtered['angle'].apply(lambda x: x.mean()) * np.pi / 180
diffs = angles.diff()


# To estimate the period, find time it takes for change in mean angle to change signs

# start_frame = data_filtered.frame[0]
# end_frame = start_frame + 1
# initial_direction = diffs[start_frame]
# while not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
#     end_frame += 1
#     print end_frame

# period = dt * (end_frame - start_frame) * 2
# disp = (angles[end_frame] - angles[start_frame]) / 2
# start_frame, end_frame = end_frame, end_frame + 1
# initial_direction = diffs[start_frame]

for frame, observations in data_filtered:
    if frame == 0:
        continue
    if frame % 100 == 0:
        print frame
    
    indices = observations.index.values
    observation_dicts = []

    observations['rank'] = observations['fol_y'].rank()


    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly, angle, rank = observation.pixlen, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y, observation.angle, observation['rank']
        angle *= np.pi / 180
        z = np.array([tipx, tipy, folx, foly, pixlen, rank])
        omega = ((diffs[frame]) / dt)

        x0 = np.array(
            [tipx, tipy, folx, foly, pixlen, omega, rank]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : (angle, folx, foly)
        })

    labels = tracker.detect(observation_dicts)
    data.loc[indices, 'ordinal'] = labels

dat['color_group'] = data['ordinal']
# Now pickle mwe and run validate_classification_results on it
data.to_pickle(output_filename)


