import numpy as np
from scipy.optimize import linear_sum_assignment


class ExtendedKalmanThread(object):
  """
    An implemention of the Kalman algorithm
    An estimation is a juxtaposed with an observation at each time step
  """
  def __init__(self, t, x0, f, F, h, H, P0=None, Q=0, R=0):
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

    if P0 == None:
      P0 = np.eye(x0.shape[0])


    self.Q, self.R, self.f, self.F, self.h, self.H,  =  Q, R, f, F, h, H
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
    x, P, Q, h, H, R, k, f, F = self.x, self.P, self.Q, self.h, self.H, self.R, self.k, self.f, self.F

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


class KalmanTracker(object):
  def __init__(self, P0, Q, R, state_factory, sensor_factory):
    self.predictors = [] #List of KalmanThreads
    self.current_predictor_label = 1
    # {'label': 1, 'predictor': KalmanThread}
    self.strikes = []

    self.P0, self.Q, self.R = P0, Q, R
    self.state_factory = state_factory
    self.sensor_factory = sensor_factory

    self.k = 0


  def detect(self, observations):

    # observations_dict:
    # {
    #   "x": np.array[[46, 34]]
    #   "z": np.array[[43]]
    #   "sensor_factory_args": {"L": 67}
    #   "state_factory_args": {"dt": 4, "period": 4}
    # }

    #Remove a predictor if it has had too many erroneous walks
    for i, strikes in enumerate(self.strikes):
      if strikes > 5:
        del self.strikes[i]
        del self.predictors[i]

    #update each prediction and form a cost matrix
    cost_matrix = np.zeros((len(observations), len(self.predictors)))

    #rows of cost matrix represent observations, columns represent predictions
    for i, observation_dict in enumerate(observations):
      for j, predictor in enumerate(self.predictors):
        z = observation_dict['z']
        _, _, _, prediction = predictor['predictor'].update_preview(z)
        cost_matrix[i, j] = np.linalg.norm(prediction - z)


    observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)

    for i in range(len(observation_indices)):
      observation_index = observation_indices[i]
      prediction_index = prediction_indices[i]

      observation_dict = observations[observation_index]
      z = observation_dict['z']

      self.predictors[prediction_index]['predictor'].update(z)



    # Prepare to add new observation if number exceeds predictions
    if len(observations) > len(self.predictors):
      mask = np.in1d(np.arange(len(observations)), observation_indices)

      unused_indices = np.where(~mask)[0]
      for i in unused_indices:  
        observation_dict = observations[i]
        x0 = observation_dict["x"]
        self.addPredictor(
          1000, x0,
          observation_dict["state_factory_args"],
          observation_dict["sensor_factory_args"],
        )

    #If cost of assignment is too high, increment strikes 
    for i, row in enumerate(cost_matrix):
      for j, cost in enumerate(row):
        if cost > 50:
          self.strikes[j] += 1

    # Return the label (numerical) of each assignment
    labels = [self.predictors[i]['label'] for i in prediction_indices]

    return labels

  def addPredictor(self, t, x0, state_factory_args=[], sensor_factory_args=[]):
    f, F = self.state_factory(*state_factory_args)
    h, H = self.sensor_factory(*sensor_factory_args)

    k = ExtendedKalmanThread(
      t,
      x0,
      f=f, F=F, h=h, H=H,
      P0=self.P0, Q=self.Q, R=self.R,
    )

    self.predictors.append(
      {
        'predictor' : k,
        'label' : self.current_predictor_label
      }
    )

    self.strikes.append(0)

    self.current_predictor_label += 1




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


P0 = np.eye(2) * 0.5
Q = np.eye(2) * 5
R = np.eye(2) * 10

tracker = KalmanTracker(
  P0, Q, R,
  create_state_functions,
  create_sensor_functions,
)

    # observations_dict:
    # {
    #   "x": np.array[[46, 34]]
    #   "z": np.array[[43]]
    #   "sensor_factory_args": {"L": 67}
    #   "state_factory_args": {"dt": 4, "period": 4}
    # }

observation = [{
  "x" : np.array([[np.pi, np.pi / 2 ]]).T,
  "z" : np.array([[55, 66]]).T,
  "sensor_factory_args" : [15],
  "state_factory_args": [0.15, 25],
}]
print tracker.detect(observation)
print tracker.detect(observation)
print tracker.detect(observation)








