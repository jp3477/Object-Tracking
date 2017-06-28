import numpy as np
import matplotlib.pyplot as plt
import pandas
import pdb
from filters import UnscentedKalmanTracker



def state_function(state, dt, angle, new_folx, new_foly):
    tipx, tipy, folx, foly, pixlen, omega = state[0], state[1], state[2], state[3], state[4], state[5]
    theta = -1 * np.arctan((tipy - foly) / (tipx - folx))

    # omega = (angle - theta) / dt
    v = omega * pixlen


    tipx = tipx + v * np.sin(theta) * dt
    tipy = tipy + v * np.cos(theta) * dt

    return np.array([tipx, tipy, folx, foly, pixlen, omega])

def measurement_function(state):
    tipx, tipy, folx, foly, pixlen = state[0], state[1], state[2], state[3], state[4]

    return np.array([tipx, tipy, folx, foly, pixlen])



#State is [tipx, tipy, folx, foly, pixlen, omega]
dim_x = 6
dim_z = 5

# F = np.eye(dim_x)
# F = np.array([
#     [1, 0,  0,  0,  ]
# ])
# H = np.eye(dim_z)

# P0 = np.eye(dim_x) * 2
# R = np.eye(dim_z)
# Q = np.eye(dim_x)

# P0 = np.eye(dim_x) * np.array([1, 1, 1, 0.5, 0.0001,])
# R = np.eye(dim_z) * np.array([1, 10, 0.0001, 0.0001, 1]) * 100000
# Q = np.eye(dim_x) * np.array([1, 1, 0.0001, 0.0001, 0.0001,]) * 0.0001

P0 = np.eye(dim_x) * np.array([1, 1, 1, 0.5, 0.0001, 0.1])
# R = np.eye(dim_z) * np.array([500, 500, 0.001, 0.001, 500,])
R = np.eye(dim_z) * np.array([70, 70, 30, 30, 10])
Q = np.eye(dim_x) * np.array([11, 11, 9, 1.8 , 44, 30])

dt = 200 ** -1

# tracker = KalmanTracker(
#     P0=P0, Q=Q, R=R, F=F, H=H,
# )
tracker = UnscentedKalmanTracker(P0, Q, R, state_function, measurement_function, dt)
whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']
predictions = {}



data = pandas.read_pickle('15000_frames_revised.pickle')
data = data[(data.frame > 10000) & (data.frame < 11000) ]

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
    if frame == 10001:
        continue
    if frame % 1000 == 0:
        print frame
    
    indices = observations.index.values
    observation_dicts = []

    for j, observation in observations.iterrows():
        pixlen, tipx, tipy, folx, foly, angle = observation.pixlen, observation.tip_x, observation.tip_y, observation.fol_x, observation.fol_y, observation.angle
        angle *= np.pi / 180
        z = np.array([tipx, tipy, folx, foly, pixlen])
        omega = ((diffs[frame]) / dt)

        # print"Calculated angle:{}\tMeasured angle:{}".format(np.arctan((tipy - foly) / (tipx - folx)), angle)

        x0 = np.array(
            [tipx, tipy, folx, foly, pixlen, omega]
        )

        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "fx_args" : (angle, folx, foly)
        })

    labels, preds = tracker.detect(observation_dicts)
    predictions[frame] = dict([(labels[i], preds[i]) for i in range(len(preds))])
    data.loc[indices, 'color_group'] = labels

    # Alter period if a sign change is observed
    # if frame <= end_frame:
    #     if not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
    #         print "changed signs!"
    #         period = dt * (end_frame - start_frame) * 2
    #         disp = (angles[end_frame] - angles[start_frame]) / 2
    #         start_frame, end_frame = end_frame, end_frame + 1
    #         initial_direction = diffs[start_frame]
    #     else:
    #         end_frame = frame


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
        # angle = whisker['angle']
        # print"Calculated angle:{}\tMeasured angle:{}".format(-1 * np.arctan((tipy - foly) / (tipx - folx)) * 180 / np.pi, angle)

        color = whisker_colors[int(whisker['color_group'])]

        plt.plot([folx, tipx], [foly, tipy], color=color)

    plt.figtext(0.4, 0.3, angles[frame] * 180 / np.pi)

    plt.pause(0.05)



