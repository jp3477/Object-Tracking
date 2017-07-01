import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import spatial

from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise


class DerivedUnscentedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points):
        UnscentedKalmanFilter.__init__(
            self, dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points, 
        )

        self.fx_args = ()

    def get_prediction(self, dt=None,  UT=None, fx_args=()):
        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i in range(self._num_sigmas):
            self.sigmas_f[i] = self.fx(sigmas[i], dt, *fx_args)

        x, P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)
        return x, P

class UnscentedKalmanTracker(object):
    """
    Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
    of observed objects
    """
    def __init__(self, P0, Q, R, f, h, dt, show_predictions=False):
        self.predictors = [] #List of KalmanThreads
        self.current_predictor_label = 1
        # {'label': 1, 'predictor': KalmanThread}
        self.strikes = []

        self.P0, self.Q, self.R, self.f, self.h, self.dt = P0, Q, R, f, h, dt
        self.show_predictions = show_predictions

        self.k = 0


    def detect(self, observations):
        #Remove a predictor if it has had too many erroneous walks
        # for i, strikes in enumerate(self.strikes):
        #   if strikes > 5:
        #     del self.strikes[i]
        #     del self.predictors[i]
        #     self.current_predictor_label -= 1

        #update each prediction and form a cost matrix

        current_predictors = self.predictors
        cost_matrix = np.zeros((len(observations), len(self.predictors)))

        #rows of cost matrix represent observations, columns represent predictions
        for i, observation_dict in enumerate(observations):
            for j, predictor in enumerate(self.predictors):
                z = observation_dict['z']
                fx_args = observation_dict['fx_args']
                x0 = predictor['predictor'].x
                x, P = predictor['predictor'].get_prediction(fx_args=fx_args)
                prediction = predictor['predictor'].hx(x)

                # dist = np.linalg.norm(prediction - z) / (640 * np.sqrt(2))
                R = predictor['predictor'].R
                dist = spatial.distance.mahalanobis(prediction, z, np.linalg.inv(R))
                detP = np.linalg.det(P)
                cost = dist
                cost_matrix[i, j] = cost

        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)
        exclusion_indices = []
        predictor_exclusion_indices = []
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]

            cost = cost_matrix[observation_index, prediction_index]
            observation_dict = observations[observation_index]
            fx_args = observation_dict['fx_args']
            z = observation_dict['z']

            self.predictors[prediction_index]['predictor'].predict(fx_args=fx_args)
            self.predictors[prediction_index]['predictor'].update(z)


        observation_indices = np.delete(observation_indices, exclusion_indices)
        prediction_indices = np.delete(prediction_indices, exclusion_indices)
        current_predictors = np.delete(current_predictors, predictor_exclusion_indices)




            # preds.append(self.predictors[prediction_index]['predictor'].hx(self.predictors[i]['predictor'].x))

        # if len(self.predictors) > len(observations):
        #   mask = np.in1d(np.arange(len(self.predictors)), prediction_indices)

        #   unused_indices = np.where(~mask)[0]

        #   for i in unused_indices:
        #     # self.predictors[i]['predictor'].Q = np.eye(5) * np.array([1, 10, 0.0001, 0.0001, 1]) * 5
        #     # self.predictors[i]['predictor'].R = np.eye(5) * np.array([1, 10, 0.0001, 0.0001, 1]) * 100000000
        #     self.predictors[i]['predictor'].predict(fx_args=self.predictors[i]['predictor'].fx_args)
        #     best_observation_index = np.argmin(cost_matrix[:, i])
        #     observation_dict = observations[best_observation_index]

        #     z = observation_dict['z']
        #     # self.predictors[i]['predictor'].update(z)


        # Prepare to add new observation if number exceeds predictions
        if len(observations) > len(prediction_indices):

            mask = np.in1d(np.arange(len(observations)), observation_indices)
            unused_indices = np.where(~mask)[0]
            for i in unused_indices:  
                observation_dict = observations[i]
                x0 = observation_dict['x']
                fx_args = observation_dict['fx_args']
                self.addPredictor(x0, fx_args)
                prediction_indices = np.append(prediction_indices, i)

        #If cost of any assignment is too high, increment strikes...threshold is arbitrary 
        for i, j in zip(observation_indices, prediction_indices):
            cost = cost_matrix[i, j]

            if cost > 50:
                self.strikes[j] += 1
            else:
                self.strikes[j] = 0

        # Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]

        if self.show_predictions:
            preds = [self.predictors[i]['predictor'].hx(self.predictors[i]['predictor'].x) for i in prediction_indices]
            return labels, preds
        else:
            return labels

    def addPredictor(self, x0, fx_args):
        dim_x = self.Q.shape[0]
        dim_z = self.R.shape[0]
        # points_class = JulierSigmaPoints(dim_x, 0)
        points_class = MerweScaledSigmaPoints(dim_x, 1e-3, 2, 0)

        kf = DerivedUnscentedKalmanFilter(
            dim_x=dim_x, dim_z=dim_z,
            dt=self.dt, hx=self.h, fx=self.f,
            points=points_class,
        )
        kf.x = x0
        kf.P = self.P0
        kf.R = self.R
        kf.Q = self.Q
        kf.fx_args = fx_args

        self.predictors.append(
          {
            'predictor' : kf,
            'label' : self.current_predictor_label
          }
        )

        self.strikes.append(0)

        self.current_predictor_label += 1

class KalmanTracker(object):
    """
    Uses Kalman Filters and assignments with a Hungarian algorithm to keep track
    of observed objects
    """
    def __init__(self, P0, F, H, Q, R , max_object_count=8, max_strikes=20, show_predictions=False):
        self.predictors = [] #List of KalmanThreads
        self.current_predictor_labels = range(max_object_count, 0, -1)
        # {'label': 1, 'predictor': KalmanThread}
        self.max_strikes = max_strikes
        self.strikes = []

        self.P0, self.Q, self.R, self.F, self.H = P0, Q, R, F, H
        self.show_predictions = show_predictions

        self.k = 0


    def detect(self, observations):
        #Remove a predictor if it has had too many erroneous walks
        current_predictors = self.predictors
        keep_indices = []
        filtered_predictors = []
        filtered_strikes = []
        for i, strikes in enumerate(self.strikes):
            if strikes < self.max_strikes:
                keep_indices.append(i)
            else:
                self.current_predictor_labels.append(i)

                # print len(self.strikes)
                # del self.strikes[i]
                # del self.predictors[i]
                # self.current_predictor_labels.append(i)

        for i in keep_indices:
            filtered_predictors.append(self.predictors[i])
            filtered_strikes.append(self.strikes[i])

        self.predictors = filtered_predictors
        self.strikes = filtered_strikes



        #update each prediction and form a cost matrix


        cost_matrix = np.zeros((len(observations), len(self.predictors)))

        #rows of cost matrix represent observations, columns represent predictions
        for i, observation_dict in enumerate(observations):
            for j, predictor in enumerate(self.predictors):
                z = observation_dict['z']

                x, P = predictor['predictor'].get_prediction()
                prediction = np.dot(predictor['predictor'].H, x)
                # R = predictor['predictor'].R
                R = np.eye(4) * np.array([5, 5, 40, 0.004])
                dist = spatial.distance.mahalanobis(prediction, z, np.linalg.inv(R))

                # dist = np.linalg.norm(x - z)
                # detP = np.linalg.det(P)
                cost = dist
                # cost = dist * detP
                cost_matrix[i, j] = cost
        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)

        exclusion_indices = []
        predictor_exclusion_indices = []

        extreme_cost_dict = {}
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]

            cost = cost_matrix[observation_index, prediction_index]

            if cost < 30:
                observation_dict = observations[observation_index]
                z = observation_dict['z']

                self.predictors[prediction_index]['predictor'].predict()
                self.predictors[prediction_index]['predictor'].update(z)
                self.strikes[prediction_index] = 0
            else:
                extreme_cost_dict[i] = cost


        sorted_exclusion_indices = sorted(extreme_cost_dict, key=extreme_cost_dict.get, reverse=True)
        exclusion_indices = sorted_exclusion_indices[:len(self.predictors) + len(sorted_exclusion_indices) - 8]
        remainding_indices = sorted_exclusion_indices[len(self.predictors) + len(sorted_exclusion_indices) - 8:]


        for i in remainding_indices:
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]  
                      
            observation_dict = observations[observation_index]
            z = observation_dict['z']

            self.predictors[prediction_index]['predictor'].predict()
            self.predictors[prediction_index]['predictor'].update(z) 
            self.strikes[prediction_index] = 0           




        # Prepare to add new observation if number exceeds predictions
        if len(observations) > len(prediction_indices):
            mask = np.in1d(np.arange(len(observations)), observation_indices)

            unused_indices = np.where(~mask)[0]
            for i in unused_indices:  
                # if len(current_predictors) < 8:
                observation_dict = observations[i]
                x0 = observation_dict["x"]
                self.addPredictor(x0)
                prediction_indices = np.append(prediction_indices, i)

        if len(current_predictors) > len(observations):
            mask = np.in1d(np.arange(len(current_predictors)), prediction_indices)

            unused_indices = np.where(~mask)[0]
            #If assignment not made for a while, increment strikes...threshold is arbitrary 

            for i in unused_indices:
                self.strikes[j] += 1

        # print self.current_predictor_labels


        # Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]

        if self.show_predictions:
            preds = [np.dot(self.predictors[i]['predictor'].H, self.predictors[i]['predictor'].x) for i in prediction_indices]
            return labels, preds
        else:
            return labels

    def addPredictor(self, x0):
        dim_x = self.Q.shape[0]
        dim_z = self.R.shape[0]

        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        kf.x = x0
        kf.F = self.F
        kf.H = self.H
        kf.P = self.P0
        kf.R = self.R
        kf.Q = self.Q

        self.predictors.append(
          {
            'predictor' : kf,
            'label' : self.current_predictor_labels.pop()
          }
        )

        self.strikes.append(0)

        # self.current_predictor_label += 1

def max_mahalanobis_distance(max_deviation, R):
    origin = np.zeros(max_deviation.shape)
    max_distance = spatial.distance.mahalanobis(max_deviation, origin, np.linalg.inv(R))

    return max_distance



