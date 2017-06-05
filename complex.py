from kalman_classes import ExtendedKalmanThread
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class StalledTurner(object):
  """ Class to test Kalman ability to distinguish objects """
  def __init__(self, theta_max, period, phi=0, t = 0, dt = 0.15 ):
    self.theta_max, self.period, self.phi, self.t, self.dt  = theta_max, period, phi, t, dt

  def turn(self):
    if self.t >= self.dt * 30 and self.t <= self.dt * 110:
      theta = self.theta_max * np.sin((2 * np.pi * self.dt * 30 / self.period) + self.phi)
    else:
      theta = self.theta_max * np.sin((2 * np.pi * self.t / self.period) + self.phi)
    
    self.t += self.dt
    return theta


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

    # if theta + omega * dt < 0:
    #   theta_new *= -1

    omega_new = omega + (-1 * (2 * np.pi / period) ** 2) * dt * theta
    # print (-1 * (2 * np.pi / period) ** 2) * dt * theta * 180 / np.pi

    return np.array([[theta_new, omega_new]]).T
  def state_function_jacobian(state):
    A = np.array([
      [1,       dt], 
      [(-1 * (2 * np.pi / period) ** 2) * dt, 1 ],
    ])
    return A

  return state_function, state_function_jacobian


t_max = 150
dt = 0.05


P0 = np.eye(2) * 0.5

# Q = np.eye(2) * 25
Q = np.array([
  [1000000,   0],
  [0,   1000000]
])
R = np.eye(2) * 0.00005

u = np.array([[0, 0]])
B = np.eye(2)

turners = [
  StalledTurner(2 * np.pi, 10, phi=0, dt=dt),
  StalledTurner(2 * np.pi, 10, phi=0, dt=dt),
  StalledTurner(2 * np.pi, 10, phi=0, dt=dt),
]
L = [35, 35, 35]
turner_points = np.zeros((2, t_max, 3))
turner_threads = []
strikes = [0, 0, 0]

for j, turner in enumerate(turners):
  theta0 = turner.theta_max * np.sin(turner.phi)
  omega0 = (2 * np.pi * turner.theta_max / turner.period) * np.cos(turner.phi)
  #x0  should be a column
  x0 = np.array([[theta0, omega0]]).T
  h, H = create_sensor_functions(L[j])
  k = ExtendedKalmanThread(t_max, x0, P0, Q, R, h, H, u, B=0)
  turner_threads.append(k)

  for i in range(t_max):
    theta = turner.turn()
    turner_points[:, i, j] = np.array([L[j] * np.cos(theta), L[j] * np.sin(theta)])

  turners[j] = turner



for i in range(1, t_max):
  cost_matrix = np.zeros((len(turner_threads), len(turner_threads)))
  for j in range(len(turners)):
    turner = turners[j]
    # A = np.array([[1,dt], [ (-1 * (2 * np.pi / turner.period) ** 2) * dt, 1 ]])
    f, F = create_state_functions(dt, turner.period)
    z = np.array([turner_points[:, i, j]]).T

    for k in range(len(turner_threads)):
      thread = turner_threads[k]
      thread.set_state_functions((f, F))
      x, P, detP, soln = thread.update_preview(z)
      #Assign cost matrix based on distance between prediction and observation
      cost_matrix[k, j] = np.linalg.norm(z - soln)

  #row_ind corresponds to estimations and col_ind refers to observations
  row_indices, col_indices = linear_sum_assignment(cost_matrix)

  for j in range(len(row_indices)):
    if row_indices[j] != col_indices[j]:
      if strikes[col_indices[j]] < 4:
        thread_index, turner_index = col_indices[j], col_indices[j]
        turner = turners[turner_index]

        z = np.array([turner_points[:, i, turner_index]]).T
        turner_threads[thread_index].set_state_functions(create_state_functions(dt, turner.period))
        soln = turner_threads[thread_index].update(z)

        strikes[thread_index] += 1
        print strikes
      else:
        print "we have a problem"
    else:        
      thread_index = row_indices[j]
      turner_index = col_indices[j]

      turner = turners[turner_index]

      z = np.array([turner_points[:, i, turner_index]]).T
      turner_threads[thread_index].set_state_functions(create_state_functions(dt, turner.period))
      soln = turner_threads[thread_index].update(z)

      strikes[thread_index] = 0



plt.ion()

for i in range(t_max):
  plt.clf()
  plt.xlim(-40, 40)
  plt.ylim(-40, 40)
  plt.title('Time: {}'.format(i))

  # print turner_threads[2].x[:, i]

  plt.plot(turner_points[0, i, 0], turner_points[1, i, 0], 'bo', markersize=10 )
  plt.plot(turner_points[0, i, 1], turner_points[1, i, 1], 'ro', markersize=10 )
  plt.plot(turner_points[0, i, 2], turner_points[1, i, 2], 'go', markersize=10 )


  plt.plot(turner_threads[0].z[0, i], turner_threads[0].z[1, i], 'ko')
  plt.plot(turner_threads[1].z[0, i], turner_threads[1].z[1, i], 'mo')
  plt.plot(turner_threads[2].z[0, i], turner_threads[2].z[1, i], 'yo')
  plt.pause(0.05)