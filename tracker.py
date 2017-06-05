import numpy as np
import matplotlib.pyplot as plt

from kalman_classes import KalmanTracker

def create_sensor_functions(L):
  def sensor_function(state):
    theta = state[0][0]
    return np.array([[L * np.cos(theta), L * np.sin(theta)]]).T
  def sensor_function_jacobian(state):
    theta = state[0][0]
    return np.array([
      [-L * np.sin(theta), 0],
      [L * np.cos(theta), 0],
    ])

  return sensor_function, sensor_function_jacobian

def create_state_functions(dt, period):
  def state_function(state):
    theta = state[0]
    omega = state[1]
    theta_new = theta + omega * dt


    omega_new = omega + (-1 * (2 * np.pi / period) ** 2) * dt * theta

    return np.array([[theta_new, omega_new]]).T
  def state_function_jacobian(state):
    A = np.array([
      [1,       dt], 
      [(-1 * (2 * np.pi / period) ** 2) * dt, 1 ],
    ])
    return A

  return state_function, state_function_jacobian

class Turner(object):
  """ Class to test Kalman ability to distinguish objects """
  def __init__(self, theta_initial, theta_max, period, phi=0, t = 0, dt = 0.15 ):
    self.theta_initial, self.theta_max, self.period, self.phi, self.t, self.dt  = theta_initial, theta_max, period, phi, t, dt

  def turn(self):
    theta = self.theta_max * np.sin((2 * np.pi * self.t / self.period) + self.phi) + self.theta_initial
    omega = (2 * np.pi * self.theta_max / self.period) * np.cos(self.phi)
    self.t += self.dt
    return theta, omega #+ (2 * np.pi/180) * np.random.randn()

t_max = 100
dt = 0.05

P0 = np.eye(2) * 10
Q = np.eye(2) * 100
R = np.eye(2) * 2

turners = [
  Turner(0, np.pi / 2, 1, phi=0, dt=dt),
  Turner(np.pi / 32, np.pi / 2, 1, phi=0, dt=dt),
  Turner(np.pi / 64, np.pi / 2, 1, phi=0, dt=dt),
]

L = [24, 35, 35]

tracker = KalmanTracker(
  P0, Q, R,
  create_state_functions,
  create_sensor_functions,
)

colors = ['r', 'b', 'g', 'c', 'k']

plt.ion()
for i in range(t_max):

  observations = []
  for j, turner in enumerate(turners):
    theta, omega = turner.turn()
    #x  should be a column
    x = np.array([[theta, omega]]).T
    z = np.array([[L[j] * np.cos(theta) + 1 * np.random.randn(), L[j] * np.sin(theta) + 1 * np.random.randn()]]).T
    # z = np.array([[L[j] * np.cos(theta), L[j] * np.sin(theta)]]).T
    sensor_factory_args = [L[j]]
    state_factory_args = [dt, turner.period]

    observations.append({
      "x": x,
      "z": z,
      "sensor_factory_args": sensor_factory_args,
      "state_factory_args": state_factory_args,
    })


  labels = tracker.detect(observations)
  print labels

  plt.clf()
  plt.xlim(-40, 40)
  plt.ylim(-40, 40)
  plt.title('Time: {}'.format(i))
  for j in range(len(labels)):
    x, y = observations[j]['z']
    color = colors[labels[j] - 1]
    plt.plot(x, y, '{}o'.format(color))



  plt.pause(0.05)







