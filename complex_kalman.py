import numpy as np
import matplotlib.pyplot as plt
import pandas
import pdb
from filters import UnscentedKalmanTracker, KalmanTracker



def state_function(state, dt):
    pixlen, rank = state[0], state[1]

    return np.array([pixlen, rank])

def measurement_function(state):
    pixlen, rank = state[0], state[1]

    return np.array([pixlen, rank])

def arctan(y, x):
    angle = np.arctan2(y, x)

    if angle > 0:
        angle += -np.pi / 2
    if angle < 0:
        angle += np.pi / 2

    return angle



#State is [tipx, tipy, pixlen]
#Measurement is [tipx, tipy]
dim_x = 5
dim_z = 2


P0 = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1])
R = np.eye(dim_z) * np.array([0.1, 0.1])
Q = np.eye(dim_x) * np.array([0.1, 0.1, 0.1, 0.1, 0.1])

F = np.eye(dim_x)
H = np.array([
    [ 1, 0, 0, 0, 0],
    [ 0, 1, 0, 0, 0],
])

dt = 200 ** -1

tracker = KalmanTracker(P0, F, H, Q, R, show_predictions=True)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}



data = pandas.read_pickle('15000_frames_revised.pickle')
data['color_group'] = 0
data = data[(data.frame > 10000) & (data.frame < 10500) ]

oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus
data.loc[data.tip_y < oof_y_thresh, 'length'] += oof_y_bonus

data_filtered = data[data.pixlen > 20].groupby('frame')
angles = data_filtered['angle'].apply(lambda x: x.mean()) * np.pi / 180
diffs = angles.diff()




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
    mean_foly = observations['fol_y'].mean()
    mean_folx = observations['fol_x'].mean()
    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly = observation.length, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y

        angle = -1 * arctan(tipy - mean_foly, tipx - mean_folx)
        z = np.array([tipx, tipy])

        x0 = np.array(
            [tipx, tipy, folx, foly, pixlen]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : ()
        })


    labels, preds = tracker.detect2(observation_dicts)
    predictions[frame] = dict([(labels[i], preds[i]) for i in range(len(preds))])
    data.loc[indices, 'color_group'] = labels



subset = data[ (data.frame > 10000) & (data.frame < 15000)].groupby('frame')

plt.ion()
for frame, whiskers in subset:
    if frame == 10001:
        continue
    plt.clf()
    plt.xlim(0, 640)
    plt.ylim(0, 640)
    plt.gca().invert_yaxis()
    plt.title('Frame: {}'.format(frame))

    preds = predictions[frame]

    for key in preds.keys():
        z = preds[key]
        color = whisker_colors[int(key)]
        tipx, tipy  = z[0], z[1]
        plt.plot(tipx, tipy, marker='o', color=color, markersize=15)
        # plt.plot(folx, foly, marker='o', color=color, markersize=15)


    for i, whisker in whiskers.iterrows():
        folx, foly = whisker['fol_x'], whisker['fol_y']
        tipx, tipy = whisker['tip_x'], whisker['tip_y']

        color = whisker_colors[int(whisker['color_group'])]

        plt.plot([folx, tipx], [foly, tipy], color=color)

    plt.figtext(0.4, 0.3, angles[frame] * 180 / np.pi)

    plt.pause(0.001)



