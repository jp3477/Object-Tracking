import pandas
import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanTracker


def create_sensor_functions(L, folx, foly):
  def sensor_function(state):
    theta = state[0][0]
    return np.array([
        [folx - np.abs(L * np.cos(theta)), L * np.sin(theta) + foly, folx, foly]
    ]).T
  def sensor_function_jacobian(state):
    theta = state[0][0]
    return np.array([
      [L * np.sin(theta), 0],
      [L * np.cos(theta), 0],
      [0, 0],
      [0, 0],
    ])

  return sensor_function, sensor_function_jacobian

def create_state_functions(dt, period):
  def state_function(state):
    theta = state[0][0]
    omega = state[1][0]
    theta_new = theta + omega * dt


    omega_new = omega + (-1 * (2 * np.pi / period) ** 2) * dt * theta
    if np.abs(omega_new) > np.pi:
        omega_new = np.pi

    return np.array([[theta_new, omega_new]]).T
  def state_function_jacobian(state):
    A = np.array([
      [1,       dt], 
      [(-1 * (2 * np.pi / period) ** 2) * dt, 1 ],
    ])
    return A

  return state_function, state_function_jacobian



whisker_colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'pink', 'orange']

P0 = np.eye(2) * 10
Q = np.eye(2) * 0.1
R = np.eye(4) * 0.1

tracker = KalmanTracker(
    P0=P0, Q=Q, R=R,
    state_factory=create_state_functions, sensor_factory=create_sensor_functions,
)
predictions = {}

data = pandas.read_pickle('15000_frames_revised.pickle')
data = data[data.frame < 2000]
# data.loc[data.tip_y < 200, 'pixlen'] += 200

data_filtered = data[data.pixlen > 40].groupby('frame')


dt = 30 ** -1
diffs = data_filtered['angle'].apply(lambda x: x.mean()).diff()

# To estimate the period, find time it takes for change in mean angle to change signs
start_frame = 1
end_frame = 2
initial_direction = diffs[start_frame]
while not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
    end_frame += 1

period = dt * (end_frame - start_frame) * 2
start_frame, end_frame = end_frame, end_frame + 1
initial_direction = diffs[start_frame]


# Run one iteration of the Kalman Tracker for each frame
print "Shifting through frames and classifying"
for frame, observations_in_frame in data_filtered:
    if frame == 0:
        continue
    if frame % 1000 == 0:
        print frame
    # indices = [idx[1] for idx in observations_in_frame.index.values]
    indices = observations_in_frame.index.values

    observation_dicts = []
    for j, o in observations_in_frame.iterrows():


        #angular velocity (omega) is set on a frame to frame basis
        omega = (diffs[frame]) / dt 

        #Pull theta from the observed angle
        theta, omega = o.angle * np.pi / 180, omega * np.pi / 180


        length, xtip, ytip, xfol, yfol = o.pixlen, o.tip_x, o.tip_y, o.fol_x, o.fol_y


        x0 = np.array([[theta, omega]]).T
        # z is an array of the x and y coordinates of the tip. Can calculate as shown or use xtip, ytip
        z = np.array([[xtip, ytip, xfol, yfol]]).T

        #Parameters that will be passed into the Kalman Filter each iteration
        sensor_factory_args = [length, xfol, yfol]
        state_factory_args = [dt, period]

        #Keep track of observation's features in an organized way for the filter
        observation_dicts.append({
            "x" : x0,
            "z" : z,
            "sensor_factory_args": sensor_factory_args,
            "state_factory_args" : state_factory_args,
        })


    # Run the filter, and generate labels for each observation 
    labels, preds = tracker.detect(observation_dicts)
    predictions[frame] = dict([(labels[i], preds[i]) for i in range(len(preds))])

    # Alter period if a sign change is observed
    if frame <= end_frame:
        if not (np.sign(diffs[end_frame]) == np.sign(initial_direction)):
            period = dt * (end_frame - start_frame) * 2
            start_frame, end_frame = end_frame, end_frame + 1
            initial_direction = diffs[start_frame]
        else:
            end_frame = frame


subset = data[ (data.frame > 500) & (data.pixlen < 1500)].groupby('frame')

plt.ion()
for frame, whiskers in subset:
  if frame == 0:
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
    tipx, tipy, folx, foly  = z[0], z[1], z[2], z[3]
    plt.plot(tipx, tipy, marker='o', color=color, markersize=15)
    plt.plot(folx, foly, marker='o', color=color, markersize=15)


  for i, whisker in whiskers.iterrows():
    folx, foly = whisker['fol_x'], whisker['fol_y']
    tipx, tipy = whisker['tip_x'], whisker['tip_y']

    color = whisker_colors[int(whisker['color_group'])]
    
    plt.plot([folx, tipx], [foly, tipy], color=color)

  plt.pause(0.01)







