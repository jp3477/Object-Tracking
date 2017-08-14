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

tracker = KalmanTracker(P0, F, H, Q, R, show_predictions=True)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}



data = pandas.read_pickle('15000_frames_revised.pickle')
data = data[(data.frame > 10000) & (data.frame < 10500) ]

oof_y_bonus = 200
oof_y_thresh = 5
data.loc[data.tip_y < oof_y_thresh, 'pixlen'] += oof_y_bonus


data_filtered = data[data.pixlen > 20].groupby('frame')
angles = data_filtered['angle'].apply(lambda x: x.mean()) * np.pi / 180
diffs = angles.diff()

frequency_table = {}



first_frame = sorted(data_filtered.groups.keys())[0]
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


    labels, preds = tracker.detect(observation_dicts)
    predictions[frame] = dict([(labels[i], preds[i]) for i in range(len(preds))])
    data.loc[indices, 'color_group'] = labels



    for i in range(observation_count):
        label = labels[i]
        frequency_table[observation_count][label].append(i)


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



