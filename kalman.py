#Implementation of the kalman filter

# function s = kalmanf(s)
# s.x = s.A*s.x + s.B*s.u;
# s.P = s.A * s.P * s.A' + s.Q;
# % Compute Kalman gain factor:
# K = s.P * s.H' * inv(s.H * s.P * s.H' + s.R);
# % Correction based on observation:
# s.x = s.x + K*(s.z - s.H *s.x);
# s.P = s.P - K*s.H*s.P;
# end
# return

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

class KalmanThread(object):
  """
    An implemention of the Kalman algorithm
    An estimation is a juxtaposed with an observation at each time step
  """
  def __init__(self, t, x0, P0, Q, R, C, u, B = 0):
    self.B, self.u, self.Q, self.C, self.R =  B, u, Q, C, R

    nstates = x0.size

    self.x = np.zeros((nstates, t))
    self.x[:, 0] = x0

    # self.P = np.zeros(t)
    # self.P[0] = P0

    # self.detP = np.zeros(t)
    # self.detP[0] = np.linalg.det(P0)

    self.P = [P0]
    self.detP = [np.linalg.det(P0)]

    self.k = 1

  def update_preview(self, A, z):
    x, B, u, P, Q, C, R, k = self.x, self.B, self.u, self.P, self.Q, self.C, self.R, self.k
    #Prediction
    #print np.dot(self.B, self.u[:, k])
    # print u[:, k-1]
    
    x_new = np.dot(A, self.x[:, k-1]) + np.dot(B, u)
    P_new = np.dot(np.dot(A, self.P[k-1]), A.T) + Q

    #Update

    G = np.dot(
          np.dot(P_new, C.T),
          np.linalg.pinv(
            np.dot(
              np.dot(C, P_new), 
              C.T
            ) + self.R
          )
        )

    x_new = x_new + np.dot(G, z - np.dot(C, x_new))
    P_new = np.dot(np.eye(x_new.size) - np.dot(G, C), P_new)
    soln = np.dot(C, x_new) + Q
    detP = np.linalg.det(P_new)

    return x_new, P_new, detP, soln

  def update(self, A, z):
    x, P, detP, soln = self.update_preview(A, z)
    self.x[:, self.k] = x
    self.P.append(P)
    self.detP.append(detP)
    self.k += 1

    return soln

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

P0 = np.eye(2) * 0.2

Q = np.eye(2) * 0.2
R = np.eye(1) * 0.2

C = np.array([[1, 0]])
u = np.array([0, 0])
B = np.eye(2)

turners = [
  Turner(np.pi, 10, phi=0, dt=0.15),
  Turner(np.pi, 5, phi= np.pi/2, dt=0.15),
  Turner(np.pi, 2.5, phi=np.pi, dt=0.15),
]
turner_angles = [[],[],[]]
turner_threads = []
strikes = [0, 0, 0]

for j, turner in enumerate(turners):
  theta0 = turner.theta_max * np.sin(turner.phi)
  omega0 = (2 * np.pi * turner.theta_max / turner.period) * np.cos(turner.phi)
  x0 = np.array([theta0, omega0])

  k = KalmanThread(t_max, x0, P0, Q, R, C, u, B=0)
  turner_threads.append(k)

  for i in range(t_max):
    turner_angles[j].append(turner.turn())

  turners[j] = turner
    

for i in range(1, t_max):
  cost_matrix = np.zeros((len(turner_threads), len(turner_threads)))
  for j in range(len(turners)):
    turner = turners[j]
    A = np.array([[1,dt], [ (-1 * (2 * np.pi / turner.period) ** 2) * dt, 1 ]])
    z = turner_angles[j][i]

    for k in range(len(turner_threads)):
      thread = turner_threads[k]
      x, P, detP, soln = thread.update_preview(A, z)
      #Assign cost matrix based on distance between prediction and observation
      cost_matrix[k, j] = np.linalg.norm(z - soln)

  #row_ind corresponds to estimations and col_ind refers to observations
  row_indices, col_indices = linear_sum_assignment(cost_matrix)

  for j in range(len(row_indices)):
    if row_indices[j] != col_indices[j]:
      if strikes[col_indices[j]] < 5:
        thread_index, turner_index = col_indices[j], col_indices[j]
        turner = turners[turner_index]
        A = np.array([[1,dt], [ (-1 * (2 * np.pi / turner.period) ** 2) * dt, 1 ]])
        z = turner_angles[turner_index][i]

        soln = turner_threads[thread_index].update(A, z)

        strikes[thread_index] += 1
      else:
        pass
    else:        
      thread_index = row_indices[j]
      turner_index = col_indices[j]

      turner = turners[turner_index]
      A = np.array([[1,dt], [ (-1 * (2 * np.pi / turner.period) ** 2) * dt, 1 ]])
      z = turner_angles[turner_index][i]

      soln = turner_threads[thread_index].update(A, z)
      strikes[thread_index] = 0
      theta = x[0]

      # if thread_index == 2:
      #   print "estimated:{}\t\tobserved:{}".format(theta * 180 / np. pi, z * 180 / np.pi )



plt.plot(range(t_max), turner_threads[0].x[0, :], 'ro')
plt.plot(range(t_max), turner_angles[0], 'r--')

plt.plot(range(t_max), turner_threads[1].x[0, :], 'bo')
plt.plot(range(t_max), turner_angles[1], 'b--')

plt.plot(range(t_max), turner_threads[2].x[0, :], 'go')
plt.plot(range(t_max), turner_angles[2], 'g--')
plt.show()






# t_max = 100
# dt = 0.15
# period = 10
# theta_max = np.pi
# phi = 0

# turner = Turner(theta_max, period, phi=phi, dt=dt)
# angles = []
# for i in range(t_max):
#   angles.append(turner.turn())
#   # print "{} Degrees".format(turner.turn() * 180 / (np.pi))

# theta0 = 0
# omega0 = (2 * np.pi * theta_max / period) * np.cos(phi)
# x0 = np.array([theta0, omega0])
# P0 = np.eye(2) * 0.4

# Q = np.eye(2) * 0.4
# R = np.eye(1) * 0.4

# C = np.array([[1, 0]])
# u = np.array([0, 0]).T
# B = np.eye(2)


# k = KalmanThread(t_max, x0, P0, Q, R, C, u, B)
# data = []
# for t in range(1, t_max):
#   A = np.array([[1,dt], [ (-1 * (2 * np.pi / period) ** 2) * dt, 1 ]])
#   z = angles[t]

#   theta = k.update(A, z)[0][0]
#   data.append(theta)

# plt.plot(range(t), data, color='r')
# print len(angles)
# plt.plot(range(t+1), angles, color='b')
# plt.show()





# t_max = 50
# dt = 0.01
# theta0, omega0 = -np.pi / 2, 2 * np.pi
# x = np.zeros((2, t_max))
# x[0, 0] =  theta0
# x[1, 0] = omega0
# T = 0.5
# L = 80


# #t is a time point
# for t in range(1, t_max):
#   A = np.array([[1,dt], [ (-1 * (2 * np.pi / T) ** 2) * dt, 1 ]])
#   x[:, t] = np.dot(A, x[:, t-1].T)


# observations = np.arange(theta0, 2 * np.pi, 2 * np.pi / 360)
# observations.apply_along_axis(np.cos, 0)
# observations[0, :] = L * np.cos(observations[0, :])
# observations[1, :] = L * np.sin(observations[1, :])


# print np.random.randn(2, t_max)
# observations = observations + np.random.randn(2, t_max)

# plt.scatter(observations[0, :], observations[1, :])
# plt.show()



