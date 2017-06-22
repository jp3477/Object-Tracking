import numpy as np
from scipy.optimize import linear_sum_assignment


class ExtendedKalmanThread(object):
	"""
	An implemention of the Kalman algorithm
	An estimation is a juxtaposed with an observation at each time step
	"""
	def __init__(self, t, x0, f, F, h, H, P0=0, Q=0, R=0):
		"""
		  t: timesteps to intialize
		  x0: column of initial state values
		  P0: initial covariance matrix of estimation process
		  Q: covariance matrix of process noise
		  R: covariances matrix of sensors
		  f: state function
		  F: Jacobian of state function
		  h: sensor function
		  H: Jacobian of sensor function
		"""


		self.P0, self.Q, self.R, self.f, self.F, self.h, self.H,  = P0, Q, R, f, F, h, H
		#x0 should be a column with size number of states
		# assert x0.shape == (2, 1)
		# assert h(x0).shape == (2, 1)
		# assert P0.shape == (2, 2)
		# assert Q.shape == (2, 2)
		# assert R.shape == (2, 2)

		nstates = x0.size
		# assert nstates == 2
		nsensors = R.shape[0]

		# self.x = np.zeros((nstates, t))
		# self.x[:, 0] = x0.T


		# self.P = [P0]
		# self.detP = [np.linalg.det(P0)]

		# self.z = np.zeros((nsensors, t))
		# self.z[:, 0] = h(x0).T

		self.x = x0
		self.P = P0
		self.detP = np.linalg.det(P0)
		self.z = h(x0)



		self.k = 1

	def predict(self, f, F, h):
		x, P, Q  = self.x, self.P, self.Q

		x_new = f(x)
		F_res = F(x)
		P_new = np.dot(np.dot(F_res, self.P), F_res.T) + Q

		return h(x_new), x_new, P_new

	def update(self, f, F, h, H, z):
		"""
		  Provides a snapshot of one timestep of the algorithm
		  z: current observation values    
		"""
		R, k = self.R, self.k

		prediction, x, P = self.predict(f, F, h)

		#Update Step
		H_res = H(x)

		G = np.dot(
			  np.dot(P, H_res.T),
			  np.linalg.pinv(
				np.dot(
				  np.dot(H_res, P), 
				  H_res.T
				) + R
			  )
			)


		x = x + np.dot(G, z - prediction)

		P = np.dot(np.eye(x.size) - np.dot(G, H_res), P)
		detP = np.linalg.det(P)

		self.x = x
		self.P = P
		self.detP = detP
		self.z = prediction
		self.k += 1

		return x, P, detP, prediction


	def set_state_functions(self, f, F):
		"""
		  Sets state-defining functions
		  setter: Two item tuple with the state function and its Jacobian
		"""
		self.f, self.F = f, F


class KalmanTracker(object):
	"""
	Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
	of observed objects
	"""
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
		# print [predictor['label'] for predictor in self.predictors]
		#Remove a predictor if it has had too many erroneous walks
		# for i, strikes in enumerate(self.strikes):
		#   if strikes > 5:
		#     del self.strikes[i]
		#     del self.predictors[i]
		#     self.current_predictor_label -= 1

		#update each prediction and form a cost matrix


		cost_matrix = np.zeros((len(observations), len(self.predictors)))

		#rows of cost matrix represent observations, columns represent predictions
		for i, observation_dict in enumerate(observations):
			for j, predictor in enumerate(self.predictors):
				z = observation_dict['z']
				state_factory_args, sensor_factory_args = observation_dict["state_factory_args"], observation_dict["sensor_factory_args"]

				f, F = self.state_factory(*state_factory_args)
				h, H = self.sensor_factory(*sensor_factory_args)


				prediction, _, _ = predictor['predictor'].predict(f, F, h)
				dist = np.linalg.norm(prediction - z)
				# print dist * detP
				# detP_weight = -1 * np.log10(detP)
				# print detP
				cost_matrix[i, j] = dist

		observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)

		for i in range(len(observation_indices)):
			observation_index = observation_indices[i]
			prediction_index = prediction_indices[i]

			# cost = cost_matrix[observation_index, prediction_index]
			# print cost
			# if cost < 50:
			observation_dict = observations[observation_index]
			z = observation_dict['z']

			state_factory_args, sensor_factory_args = observation_dict["state_factory_args"], observation_dict["sensor_factory_args"]
			f, F = self.state_factory(*state_factory_args)
			h, H = self.sensor_factory(*sensor_factory_args)

			self.predictors[prediction_index]['predictor'].update(f, F, h, H, z)


		# Prepare to add new observation if number exceeds predictions
		if len(observations) > len(self.predictors):
			mask = np.in1d(np.arange(len(observations)), observation_indices)

			unused_indices = np.where(~mask)[0]
			for i in unused_indices:  
				observation_dict = observations[i]
				x0 = observation_dict["x"]
				self.addPredictor(
				  1000000, x0,
				  observation_dict["state_factory_args"],
				  observation_dict["sensor_factory_args"],
				)
				prediction_indices = np.append(prediction_indices, i)


		#If cost of any assignment is too high, increment strikes...threshold is arbitrary 
		for i, j in zip(observation_indices, prediction_indices):
			cost = cost_matrix[i, j]
			# print "i:{}\t j:{}\t cost:{}".format(i, j, cost)

			if cost > 50:
				self.strikes[j] += 1
			else:
				self.strikes[j] = 0

		# Return the label (numerical) of each assignment
		labels = [self.predictors[i]['label'] for i in prediction_indices]
		preds = [self.predictors[i]['predictor'].z for i in prediction_indices]

		return labels, preds

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





# observations_dict:
# {
#   "x": np.array[[46, 34]]
#   "z": np.array[[43]]
#   "sensor_factory_args": {"L": 67}
#   "state_factory_args": {"dt": 4, "period": 4}
# }







