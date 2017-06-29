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



#State is [pixlen, rank]
dim_x = 2
dim_z = 2


P0 = np.eye(dim_x) * np.array([0.0001, 0.0001])
# R = np.eye(dim_z) * np.array([500, 500, 0.001, 0.001, 500,])
R = np.eye(dim_z) * np.array([10000, 10000])
Q = np.eye(dim_x) * np.array([25, 0.25])

F = np.eye(dim_x)
H = np.eye(dim_z)

dt = 200 ** -1

tracker = UnscentedKalmanTracker(P0, Q, R, state_function, measurement_function, dt, show_predictions=False)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}


# Load the data to classify
data = pandas.read_pickle(os.path.expanduser(
    '/mnt/nas2/homes/chris/whisker/test_bed/161215_KM91/masked_whisker_ends'))
data = data[(data.frame > 0) & (data.frame < 12000)].copy()

data['ordinal'] = 0

oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus

data_filtered = data[data.pixlen > 20].groupby('frame')



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

first_frame = data_filtered.groups.keys()[0]
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

        z = np.array([pixlen, rank])


        x0 = np.array(
            [pixlen, rank]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : ()
        })

    labels = tracker.detect(observation_dicts)
    data.loc[indices, 'color_group'] = labels

dat['color_group'] = data['ordinal']
# Now pickle mwe and run validate_classification_results on it
data.to_pickle(output_filename)


