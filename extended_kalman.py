import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class ExtendedKalmanThread(object):
  """
    An implemention of the Kalman algorithm
    An estimation is a juxtaposed with an observation at each time step
  """
  def __init__(self, t, x0, P0, Q, R, h, H, u, B = 0):
    """
      t: timesteps
      x0: column of initial state values
      P0: initial covariance matrix of estimation process
      Q: covariance matrix of process noise
      R: covariances matrix of sensors
      h: sensor function
      H: Jacobian of sensor function
      u: control function
      B: scale of control function
    """

    self.B, self.u, self.Q, self.h, self.H, self.R =  B, u, Q, h, H, R
    #x0 should be a column with size number of states
    assert x0.shape == (2, 1)
    assert h(x0).shape == (2, 1)
    assert P0.shape == (2, 2)
    assert Q.shape == (2, 2)
    assert R.shape == (2, 2)

    nstates = x0.size
    assert nstates == 2
    nsensors = R.shape[0]

    self.x = np.zeros((nstates, t))
    self.x[:, 0] = x0.T


    self.P = [P0]
    self.detP = [np.linalg.det(P0)]

    self.z = np.zeros((nsensors, t))
    self.z[:, 0] = h(x0).T



    self.k = 1

  def update_preview(self, z):
    """
      Provides a snapshot of one timestep of the algorithm
      z: current observation values    
    """
    x, B, u, P, Q, h, H, R, k, f, F = self.x, self.B, self.u, self.P, self.Q, self.h, self.H, self.R, self.k, self.f, self.F

    #Prediction Step
    x_new = f(self.x[:, k-1])
    assert x_new.shape == (2, 1)
    F_res = F(self.x[:, k-1])
    assert(F_res).shape == (2, 2)
    P_new = np.dot(np.dot(F_res, self.P[k-1]), F_res.T) + Q
    assert(P_new).shape == (2, 2)


    #Update Step
    H_res = H(x_new)
    h_res = h(x_new)
    assert(H_res).shape == (2, 2)
    assert(h_res).shape == (2, 1)

    G = np.dot(
          np.dot(P_new, H_res.T),
          np.linalg.pinv(
            np.dot(
              np.dot(H_res, P_new), 
              H_res.T
            ) + R
          )
        )
    
    assert z.shape == h_res.shape

    x_new = x_new + np.dot(G, z - h_res)

    P_new = np.dot(np.eye(x_new.size) - np.dot(G, H_res), P_new)
    soln = h_res
    detP = np.linalg.det(P_new)

    return x_new, P_new, detP, soln

  def update(self, z):
    """
      Runs one time step of the algorithm
      z: current observation values
    """
    x, P, detP, soln = self.update_preview(z)
    self.x[:, self.k] = x.T
    self.P.append(P)
    self.detP.append(detP)
    self.z[:, self.k] = soln.T
    self.k += 1

    return soln

  def set_state_functions(self, setter):
    """
      Sets state-defining functions
      setter: Two item tuple with the state function and its Jacobian
    """
    self.f, self.F = setter[0], setter[1]



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
    theta_new = (theta + omega * dt) % (2 * np.pi)
    if theta + omega * dt < 0:
      theta_new *= -1
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






class Turner(object):
  """ Class to test Kalman ability to distinguish objects """
  def __init__(self, theta_max, period, phi=0, t = 0, dt = 0.15 ):
    self.theta_max, self.period, self.phi, self.t, self.dt  = theta_max, period, phi, t, dt

  def turn(self):
    theta = self.theta_max * np.sin((2 * np.pi * self.t / self.period) + self.phi)
    self.t += self.dt
    return theta


t_max = 100
dt = 0.15


P0 = np.eye(2) * 0.4

Q = np.eye(2) * 0.4
R = np.eye(2) * 0.4

u = np.array([[0, 0]])
B = np.eye(2)

turners = [
  Turner(np.pi, 5, phi=0, dt=0.15),
  Turner(np.pi, 5, phi=0, dt=0.15),
  Turner(np.pi, 5, phi=0, dt=0.15),
]
L = [15, 20, 35]
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
      if strikes[col_indices[j]] < 5:
        thread_index, turner_index = col_indices[j], col_indices[j]
        turner = turners[turner_index]

        z = np.array([turner_points[:, i, turner_index]]).T
        turner_threads[thread_index].set_state_functions(create_state_functions(dt, turner.period))
        soln = turner_threads[thread_index].update(z)

        strikes[thread_index] += 1
      else:
        pass
    else:        
      thread_index = row_indices[j]
      turner_index = col_indices[j]

      turner = turners[turner_index]

      z = np.array([turner_points[:, i, turner_index]]).T
      turner_threads[thread_index].set_state_functions(create_state_functions(dt, turner.period))
      soln = turner_threads[thread_index].update(z)

      strikes[thread_index] = 0



# plt.plot(range(t_max), turner_threads[0].x[0, :], 'ro')
# plt.plot(range(t_max), turner_angles[0], 'r--')

# plt.plot(range(t_max), turner_threads[1].x[0, :], 'bo')
# plt.plot(range(t_max), turner_angles[1], 'b--')

# plt.plot(range(t_max), turner_threads[2].x[0, :], 'go')
# plt.plot(range(t_max), turner_angles[2], 'g--')
 
plt.plot(turner_threads[0].z[0, ::5], turner_threads[0].z[1, ::5], 'bo')
plt.plot(turner_threads[1].z[0, ::5], turner_threads[1].z[1, ::5], 'ro')
plt.plot(turner_threads[2].z[0, ::5], turner_threads[2].z[1, ::5], 'go')


# plt.plot(turner_points[0, ::5, 0], turner_points[1, ::5, 0], 'bo' )
# plt.plot(turner_points[0, ::5, 1], turner_points[1, ::5, 1], 'ro' )
# plt.plot(turner_points[0, ::5, 2], turner_points[1, ::5, 2], 'go' )
plt.show()






