import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import spatial
from scipy import stats
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform, JulierSigmaPoints
from constraints import AngleConstraint, FollicleConstraint
from intersect import closestDistanceBetweenLines

import scipy.integrate as integrate

class KalmanTracker(object):

    def __init__(self, P0, F, H, Q, R , max_object_count=8, max_strikes=20, show_predictions=False):
        """ Uses Kalman Filters and Fuzzy Logic to keep track of observed objects

            P0 : Intial prediction error
            F : Update matrix
            H : Measurement matrix
            Q : Process error covariance
            H : Measurement error covariance
            max_object_count : Maximum number of objects to track
            max_strikes : Number of allowable missing steps until an object's tracker is suspended
            show_predictions: Should predicted coordinates be returned

        """
        self.predictors = [] #List of KalmanThreads

        #Labels to assign to objects placed on a stack
        self.current_predictor_labels = range(max_object_count, 0, -1)

        self.max_strikes = max_strikes

        self.max_object_count = max_object_count

        #List to keep track of tracker strikess
        self.strikes = []

        self.P0, self.Q, self.R, self.F, self.H = P0, Q, R, F, H
        self.show_predictions = show_predictions

        #Keep track of step count
        self.k = 1



    def detect(self, observations):
        """ Match observations to existing predictors

            observations :  List of dictionaries with information about a detected observations


        """

        #Remove a predictor if it has had too many erroneous walks
        # current_predictors = self.predictors
        # keep_indices = []
        # filtered_predictors = []
        # filtered_strikes = []
        # for i, strikes in enumerate(self.strikes):
        #     if strikes < self.max_strikes:
        #         keep_indices.append(i)
        #     else:
        #         self.current_predictor_labels.append(current_predictors[i]['label'])


        # for i in keep_indices:
        #     filtered_predictors.append(self.predictors[i])
        #     filtered_strikes.append(self.strikes[i])

        # self.predictors = filtered_predictors
        # self.strikes = filtered_strikes


        #Match observations with trackers
        observation_indices, prediction_indices = [], []
        used_indices = []
        
        exclusion_count = 0
        exclusion_indices = []

        for i, observation_dict in enumerate(observations):
            # Costs of assigning each predictor to the current obvervation
            # Initialize cost of already matched predictors to infinitiy
            cost_list = np.zeros(len(self.predictors))
            cost_list[prediction_indices] = np.inf

            dist_list = np.zeros(len(self.predictors))
            dist_list[prediction_indices] = np.inf


            for j, predictor in enumerate(self.predictors):
                # Iterate through predictors that have not already been matched
                if j not in prediction_indices and j not in exclusion_indices:

                    # Use Kalman Filter prediction to find distance to observation
                    z = observation_dict['z']
                    x, P = predictor['predictor'].get_prediction()
                    prediction = np.dot(predictor['predictor'].H, x)
                    dist = np.linalg.norm(prediction - z)
                    # dist = spatial.distance.mahalanobis(prediction, z, np.linalg.inv(R))

                    likelihood = 1

                    # Compare candidate observation to already matched observations to find likelihood
                    # that predictor matches rules
                    for k, observation_index in enumerate(observation_indices):
                        prediction_index = prediction_indices[k]

                        matched_predictor = self.predictors[prediction_index]

                        # Contains rules between two predictors
                        constraints = matched_predictor['rules'][j]

                        # Define relationships between two observations
                        x1 = observations[observation_index]['x']
                        x2 = observation_dict['x']

                        x1_folx, x1_foly = x1[2], x1[3]
                        x1_tipx, x1_tipy = x1[0], x1[1]

                        x2_folx, x2_foly = x2[2], x2[3]
                        x2_tipx, x2_tipy = x2[0], x2[1]

                        # intersection = seg_intersect([x_obs_folx, x_obs_foly], [x_obs_tipx, x_obs_tipy], [x_other_folx, x_other_foly], [x_other_tipx, x_other_tipy])
                        # intersection_dist = np.linalg.norm(intersection - np.array([x_obs_folx, x_obs_foly]))
                        # _, _, dist_between_segments = closestDistanceBetweenLines([x_obs_folx, x_obs_foly, 0], [x_obs_tipx, x_obs_tipy, 0], [x_other_folx, x_other_foly, 0], [x_other_tipx, x_other_tipy, 0], clampAll=True)

                        area = area_between((x1_folx, x1_foly), (x1_tipx, x1_tipy), (x2_folx, x2_foly), (x2_tipx, x2_tipy))
                        closeness = x1[5] - x2[5]


                        length_diff = x1[4] - x2[4]
                        fol_diff = x1_foly - x2_foly
                        # abs_fol_diff = np.abs(fol_diff)
            

                        # Find congruity or how well the observation relationships match the expected constraints
                        congruity = constraints.compute_congruity(
                            {
                                # 'area_diff': area,
                                # 'overlap': dist_between_segments,
                                'closeness': closeness,
                                'length_diff': length_diff,
                                # 'closeness': abs_fol_diff,
                            }
                        )
                        # print "{}->{}\tcloseness_rule: {}\t closeness: {}\t length_rule: {}\t length_diff: {}\t congruity: {} ".format(j, prediction_index, constraints.rule_dict['closeness_rule'], closeness, constraints.rule_dict['length_rule'], length_diff, congruity)

                        #Aggregate total likelihood based on congruity of this observation with other observations
                        likelihood *= congruity

                    # Determine cost as some combination of cost and likelihood (might have to be tweaked)

                    # cost = 100 ** (-1 * np.log(likelihood) + 1) * dist
                    cost = 5 ** (-1 * np.log(likelihood) + 1)
                    # cost = -1 * np.log(likelihood) + 1
                    cost_list[j] = cost
                    dist_list[j] = dist

            if len(self.predictors) - i > 0:
                # Choose matched prediction based on lowest cost
                prediction_index = np.argmin(cost_list)
                # print cost_list[prediction_index]
                # if len(self.current_predictor_labels) - exclusion_count > 0 and dist_list[prediction_index] > 70 and cost_list[prediction_index] > 3 :
                #     # print cost_list[prediction_index], dist_list[prediction_index]
                #     exclusion_count += 1
                #     exclusion_indices.append(j)
                # else:
                observation_indices.append(i)
                prediction_indices.append(prediction_index)  


        preds = np.zeros((len(observations), 2))
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]




            predictor = self.predictors[prediction_index]['predictor']



            observation_dict = observations[observation_index]
            z = observation_dict['z']

            # preds.append(np.dot(predictor.H, predictor.x))
            if self.show_predictions:
                prediction = np.dot(predictor.H, predictor.x)# + self.R
                preds[observation_index, :] = prediction


            # Use Kalman filter to update prediction based on the observation
            # self.predictors[prediction_index]['predictor'].R = self.R
            self.predictors[prediction_index]['predictor'].predict()
            self.predictors[prediction_index]['predictor'].update(z)
            self.strikes[prediction_index] = 0


        # Add new tracker if number of observations exceeds number of trackers
        if len(observations) > len(prediction_indices):
            mask = np.in1d(np.arange(len(observations)), observation_indices)

            unused_indices = np.where(~mask)[0]
            new_prediction_indices = np.zeros(len(observations))
            new_prediction_indices[observation_indices] = prediction_indices
            for i in unused_indices:
                observation_dict = observations[i]
                x0 = observation_dict["x"]
                new_prediction_index = self.addPredictor(x0)
                new_prediction_indices[i] = new_prediction_index

            prediction_indices = new_prediction_indices.astype(int)
            self.update_constraints(observations, prediction_indices)

        if len(self.predictors) > len(observations):
            mask = np.in1d(np.arange(len(self.predictors)), prediction_indices)

            unused_indices = np.where(~mask)[0]
            #If assignment not made for a while, increment strikes...threshold is arbitrary 

            for i in unused_indices:
                self.strikes[j] += 1


        labels = [self.predictors[i]['label'] for i in prediction_indices]
        self.k += 1

        if self.show_predictions: 
            return labels, preds
        else:
            return labels



    def addPredictor(self, x0):
        """ Add a new predictor

            x0 : Array containing information about new predictor

        """
        dim_x = self.Q.shape[0]
        dim_z = self.R.shape[0]

        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        kf.x = x0
        kf.F = self.F
        kf.H = self.H
        kf.P = self.P0
        kf.R = self.R
        kf.Q = self.Q

        rules = {}

        new_index = len(self.predictors)


        self.predictors.append(
          {
            'predictor' : kf,
            'rules' : rules,
            'label' : self.current_predictor_labels.pop()
          }
        )



        self.strikes.append(0)
        return new_index

    def update_constraints(self, observations, prediction_indices):
        """ Define predictor relationships based on relationships between corresponding observation

            observations : List of dictionaries with information about detected observations
            prediction_indices : Indices of predictors (corresponds to observations)

        """
        assert len(observations) == len(prediction_indices)

        for i, obs1 in enumerate(observations):
            x1 = obs1['x']
            tipx, tipy, folx, foly, pixlen = x1[0], x1[1], x1[2], x1[3], x1[4]
            rank = x1[5]

            j = i + 1
            while j < len(observations):
                obs2 = observations[j]

                x2 = obs2['x']
                tipx_predictor, tipy_predictor, folx_predictor, foly_predictor, pixlen_predictor = x2[0], x2[1], x2[2], x2[3], x2[4]
                rank_predictor = x2[5]

                # _, _, dist_between_segments = closestDistanceBetweenLines([folx, foly, 0], [tipx, tipy, 0], [folx_predictor, foly_predictor, 0], [tipx_predictor, tipy_predictor, 0], clampAll=True)
                area = area_between((folx, foly), (tipx, tipy), (folx_predictor, foly_predictor), (tipx_predictor, tipy_predictor))
                closeness = rank - rank_predictor
                length_diff = pixlen - pixlen_predictor

                pixlen_rule = ''
                fol_rule = ''
                closeness_rule = ''
                overlap_rule = ''

                if pixlen - pixlen_predictor < 0:
                    pixlen_rule = 'shorter'
                elif pixlen - pixlen_predictor >= 0:
                    pixlen_rule = 'longer'

                if area <= 0:
                    fol_rule = 'above'
                elif area > 0:
                    fol_rule = 'below'


                if closeness <= 2 :
                    closeness_rule = 'near'
                else:
                    closeness_rule = 'far'


                # if dist_between_segments < 5:
                #     overlap_rule = 'true'
                # else:
                #     overlap_rule = 'false'

                closeness_rule = closeness
                pixlen_rule = length_diff

                # Record rules that define the relationship between these two predictors
                rules = {
                    'length_rule': pixlen_rule,
                    'fol_rule' : fol_rule, 
                    'closeness_rule' : closeness_rule,
                    'overlap_rule': overlap_rule
                }
                # print "{}->{} \t rules: {} \t segment_dist: {}".format(j, i, rules, dist_between_segments) 
                print "{}->{} \t rules: {} \t segment_area: {}".format(j, i, rules, area) 


                # Define converse rules as well
                opp_rules = {}

                if pixlen_rule == 'shorter':
                    opp_rules['length_rule'] = 'longer'
                elif pixlen_rule == 'longer':
                    opp_rules['length_rule'] = 'shorter'


                if fol_rule == 'above':
                    opp_rules['fol_rule'] = 'below'
                elif fol_rule == 'below':
                    opp_rules['fol_rule'] = 'above'


                if closeness_rule == 'near':
                    opp_rules['closeness_rule'] = 'far'
                else:
                    opp_rules['closeness_rule'] = 'near'

                opp_rules['overlap_rule'] = overlap_rule
                opp_rules['closeness_rule'] = closeness * -1
                opp_rules['length_rule'] = length_diff * -1

                prediction_index1 = prediction_indices[i]
                prediction_index2 = prediction_indices[j]

                # if j not in self.predictors[prediction_index1]['rules']:
                self.predictors[prediction_index1]['rules'][j] = FollicleConstraint(rules)
                # if i not in self.predictors[prediction_index2]['rules']:
                self.predictors[prediction_index2]['rules'][i] = FollicleConstraint(opp_rules)

                j += 1



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
    Uses Unscented Kalman Filters and assignments with a Hungarian algorithm to keep track
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

                dist = np.linalg.norm(prediction - z)
                R = predictor['predictor'].R
                # dist = spatial.distance.mahalanobis(prediction, z, np.linalg.inv(R))
                detP = np.linalg.det(P)
                cost = dist
                cost_matrix[i, j] = cost

        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)
        exclusion_indices = []
        predictor_exclusion_indices = []
        preds = []
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]

            predictor = self.predictors[prediction_index]['predictor']
            prediction = predictor.hx(predictor.x)
                
            observation_dict = observations[observation_index]
            z = observation_dict['z']
            preds.append(prediction)

            cost = cost_matrix[observation_index, prediction_index]
            observation_dict = observations[observation_index]
            fx_args = observation_dict['fx_args']
            z = observation_dict['z']

            self.predictors[prediction_index]['predictor'].predict(fx_args=fx_args)
            self.predictors[prediction_index]['predictor'].update(z)


        # observation_indices = np.delete(observation_indices, exclusion_indices)
        # prediction_indices = np.delete(prediction_indices, exclusion_indices)
        # current_predictors = np.delete(current_predictors, predictor_exclusion_indices)




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










def max_mahalanobis_distance(max_deviation, R):
    origin = np.zeros(max_deviation.shape)
    max_distance = spatial.distance.mahalanobis(max_deviation, origin, np.linalg.inv(R))

    return max_distance

def logpdf_single(value, distribution):
    mean = np.mean(distribution)
    std = np.std(distribution) + 1e-18
    return stats.norm(mean, std).logpdf(value)

def logpdf(values, distributions):
    total = 0
    for i, value in enumerate(values):
        distribution = distributions[i]

        prob = logpdf_single(value, distribution)
        total += prob


    return total

def logpmf_single(value, distribution, max_object_count):
    added = False
    if value not in distribution:
        distribution.append(value)
    # if len(distribution) == 0:
    #     distribution.append(value)
    #     added = True

    xk = np.arange(0, max_object_count)
    pk = np.bincount(distribution, minlength=max_object_count) / float(len(distribution))
    custm = stats.rv_discrete(name='custm', values=(xk, pk))

    return custm.logpmf(value)

def logpmf(values, distributions, max_object_count):
    total = 0
    for i, value in enumerate(values):
        distribution = distributions[i]

        prob = logpmf_single(value, distribution, max_object_count)
        total += prob

    return total

def area_between(f1, t1, f2, t2):
    """ Finds area between two whiskers (modeled as line segments)

    f1 : follicle coordinates of whisker 1
    t1 : tip coordinates of whisker 1
    f2 : follicle coordinates of whisker 2
    t2 : tip coordinates of whisker 2

    """
    dist1 = np.linalg.norm(np.array(f1) - np.array(t1))
    dist2 = np.linalg.norm(np.array(f2) - np.array(t2))

    limits = ()

    if dist1 < dist2:
        limits = (t1[0], f1[0])
    else:
        limits = (t2[0], f2[0])


    m1 = (t1[1] - f1[1]) / (t1[0] - f1[0])
    m2 = (t2[1] - f2[1]) / (t2[0] - f2[0])

    fn = lambda x: (m2 * (x - t2[0]) + t2[1]) - (m1 * (x - t1[0]) + t1[1])

    result = integrate.quad(fn, limits[0], limits[1])

    return result[0]


