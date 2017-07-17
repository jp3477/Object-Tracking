import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import spatial
from scipy import stats
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, unscented_transform, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise
from constraints import AngleConstraint, FollicleConstraint
from intersect import seg_intersect

# from HungarianMurty import k_best_costs






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
        self.max_object_count = max_object_count
        self.strikes = []
        self.rankings = None

        self.P0, self.Q, self.R, self.F, self.H = P0, Q, R, F, H
        self.show_predictions = show_predictions

        self.k = 1


    # def initalize_predictors(observations):
    #     observation_dict = observations[i]
    #     x0 = observation_dict["x"]
    #     new_prediction_index = self.addPredictor(x0)
    #     new_prediction_indices[i] = new_prediction_index

    #     for i, observation_dict in enumerate(observations):
    #         x0 = observation_dict["x"]

    #         self.addPredictor(x0)

    def detect2(self, observations):

        #Remove a predictor if it has had too many erroneous walks
        current_predictors = self.predictors
        keep_indices = []
        filtered_predictors = []
        filtered_strikes = []
        for i, strikes in enumerate(self.strikes):
            if strikes < self.max_strikes:
                keep_indices.append(i)
            else:
                self.current_predictor_labels.append(current_predictors[i]['label'])


        for i in keep_indices:
            filtered_predictors.append(self.predictors[i])
            filtered_strikes.append(self.strikes[i])

        self.predictors = filtered_predictors
        self.strikes = filtered_strikes

        observations_cp = observations
        predictors_cp = self.predictors

        observation_indices, prediction_indices = [], []
        used_indices = []
        
        exclusion_count = 0

        for i, observation_dict in enumerate(observations):
            cost_list = np.zeros(len(self.predictors))
            cost_list[prediction_indices] = np.inf


            for j, predictor in enumerate(self.predictors):
                if j not in prediction_indices:
                    z = observation_dict['z']
                    x, P = predictor['predictor'].get_prediction()
                    prediction = np.dot(predictor['predictor'].H, x)
                    dist = np.linalg.norm(prediction - z)

                    likelihood = 1

                    for k, observation_index in enumerate(observation_indices):
                        prediction_index = prediction_indices[k]
                        matched_predictor = self.predictors[prediction_index]
                        constraints = matched_predictor['rules'][j]


                        x_obs = observations[observation_index]['x']
                        x_other = observation_dict['x']

                        x_obs_folx, x_obs_foly = x_obs[2], x_obs[3]
                        x_obs_tipx, x_obs_tipy = x_obs[0], x_obs[1]

                        x_other_folx, x_other_foly = x_other[2], x_other[3]
                        x_other_tipx, x_other_tipy = x_other[0], x_other[1]

                        intersection = seg_intersect([x_obs_folx, x_obs_foly], [x_obs_tipx, x_obs_tipy], [x_other_folx, x_other_foly], [x_other_tipx, x_other_tipy])
                        intersection_dist = np.linalg.norm(intersection - np.array([x_obs_folx, x_obs_foly]))
                        # length_diff = x_obs[3] - x_other[3]
                        fol_diff = x_obs[3] - x_other[3]
                        # abs_fol_diff = np.abs(fol_diff)
            

                        # print constraints.rule_dict
                        # print                             {
                        #         'length_diff': length_diff,
                        #         'fol_diff': fol_diff,
                        #         'closeness': abs_fol_diff,
                        #     }
                        congruity = constraints.compute_congruity(
                            {
                                'fol_diff': fol_diff,
                                'intersection_dist': intersection_dist
                                # 'closeness': abs_fol_diff,
                            }
                        )

                        # if j == 4 and prediction_index == 1:
                            # print "{}->{}\tcongruity: {}".format(self.predictors[j]['label'], self.predictors[prediction_index]['label'], congruity)

                        likelihood *= congruity

                    # print "{}%".format((likelihood ** -1 / dist)*100 )
                    # cost = dist * likelihood ** -1
                    # print "cost: {}\tdist: {}".format(cost, dist)
                    # cost = dist
                    # cost = likelihood ** -1

                    cost = 20 ** (-1 * np.log(likelihood) + 1)
                    # print "dist: {}\t cost: {}\t likelihood: {}".format(dist, cost, likelihood)
                    cost_list[j] = cost
            if len(self.predictors) - i > 0:
                prediction_index = np.argmin(cost_list)

                if cost_list[prediction_index] > 1000 and len(self.current_predictor_labels) - exclusion_count > 0:
                    print cost_list[prediction_index]
                    exclusion_count += 1
                else:
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

            self.predictors[prediction_index]['predictor'].R = self.R
            self.predictors[prediction_index]['predictor'].predict()
            self.predictors[prediction_index]['predictor'].update(z)
            self.strikes[prediction_index] = 0


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


 








        # for i, observation_dict in enumerate(observations):
        #     cost_matrix = np.zeros((len(observations), i + 1))

        #     predictor = self.predictors[0]['predictor']
        #     z = observation_dict['z']
        #     x, P = predictor.get_prediction()
        #     prediction = np.dot(predictor['predictor'].H, x)
        #     dist = np.linalg.norm(prediction - z)
        #     cost_matrix[i, 0] = 


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
                self.current_predictor_labels.append(current_predictors[i]['label'])


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

                #Guess the output for a prediction and set its distance from observed value as cost
                x, P = predictor['predictor'].get_prediction()
                prediction = np.dot(predictor['predictor'].H, x)


                dist = np.linalg.norm(prediction - z)
                cost = dist
                cost_matrix[i, j] = cost

        #Assign observations to predictions with cost matrix
        observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)


        prelim_labels = [self.predictors[i]['label'] for i in prediction_indices]
        
        for i, label in enumerate(prelim_labels):
            #Find likelihood that a predictor is in this order with this number of observations
            likelihood = self.individual_log_likelihood(label, observation_indices[i], len(observations))
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]
            
            # if label == 2:
                # print "frame: {} \t cost: {} \t".format(self.k + 10000, cost_matrix[observation_index, prediction_index])

            #Give a bonus to extremely likely matches, and detract from unlikely matches
            if likelihood < -1.4:
                # print likelihood
                cost_matrix[observation_index, prediction_index] += 15
                # print cost_matrix[observation_index, prediction_index] - 50, cost_matrix[observation_index, prediction_index]
            elif likelihood > -0.06:
                cost_matrix[observation_index, prediction_index] -= 15
        # observation_indices, prediction_indices = linear_sum_assignment(cost_matrix)



        exclusion_indices = []
        predictor_exclusion_indices = []

        extreme_cost_dict = {}
        preds = np.zeros((len(observations), 2))
        for i in range(len(observation_indices)):
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]



            cost = cost_matrix[observation_index, prediction_index]

            if self.predictors[prediction_index]['label'] == 2:
                predictor = self.predictors[prediction_index]['predictor']
                print "frame: {} \t cost: {} \t".format(self.k + 10000, cost)
            predictor = self.predictors[prediction_index]['predictor']



            if (cost < 100 or predictor.x[2] > 250):
               
                
                observation_dict = observations[observation_index]
                z = observation_dict['z']

                # preds.append(np.dot(predictor.H, predictor.x))
                if self.show_predictions:
                    prediction = np.dot(predictor.H, predictor.x)# + self.R
                    preds[observation_index, :] = prediction

                self.predictors[prediction_index]['predictor'].R = self.R
                self.predictors[prediction_index]['predictor'].predict()
                self.predictors[prediction_index]['predictor'].update(z)
                self.strikes[prediction_index] = 0
            else:
                # extreme_cost_dict[i] = cost / self.predictors[prediction_index]['predictor'].x[2]
                extreme_cost_dict[i] = cost


        sorted_exclusion_indices = sorted(extreme_cost_dict, key=extreme_cost_dict.get, reverse=True)

        # exclusion_indices = sorted_exclusion_indices[:len(self.predictors) + len(sorted_exclusion_indices) - 8]
        # remainding_indices = sorted_exclusion_indices[len(self.predictors) + len(sorted_exclusion_indices) - 8:]

        # right_cutoff = max(0, self.max_object_count - (len(self.predictors) + len(sorted_exclusion_indices) + 1))
        # cutoff = min(len(sorted_exclusion_indices), right_cutoff)

        # # print len(self.predictors), len(sorted_exclusion_indices), cutoff
        # exclusion_indices = sorted_exclusion_indices[:cutoff]
        # remainding_indices = sorted_exclusion_indices[cutoff:]

        cutoff = min(self.max_object_count - len(self.predictors), len(sorted_exclusion_indices))
        exclusion_indices = sorted_exclusion_indices[:cutoff]
        remainding_indices = sorted_exclusion_indices[cutoff:]



        for i in remainding_indices:
            observation_index = observation_indices[i]
            prediction_index = prediction_indices[i]
            cost = cost_matrix[observation_index, prediction_index]
            # if self.predictors[prediction_index]['label'] == 6:
            #     print "cost2: {}".format(cost)
            #     print "sorted_exclusion_indices: {} \t exclusion_indices: {}\t predictor length: {} \t cutoff: {}".format(sorted_exclusion_indices, exclusion_indices, len(self.predictors), len(self.predictors))              
            if self.show_predictions:
                preds[observation_index, :] = np.dot(self.predictors[prediction_index]['predictor'].H, self.predictors[prediction_index]['predictor'].x)

            observation_dict = observations[observation_index]
            z = observation_dict['z']
            # print len(self.predictors) + len(exclusion_indices)
            # if self.predictors[prediction_index]['predictor'].x[2] < 250:
            #     self.predictors[prediction_index]['predictor'].R = np.eye(2) * np.array([1000, 1000])
            # else:
            self.predictors[prediction_index]['predictor'].R = np.eye(2) * np.array([1000, 1000])
            self.predictors[prediction_index]['predictor'].predict()
            self.predictors[prediction_index]['predictor'].update(z) 
            self.strikes[prediction_index] = 0           


        observation_indices = np.delete(observation_indices, exclusion_indices)
        prediction_indices = np.delete(prediction_indices, exclusion_indices)

        # current_predictors = np.delete(current_predictors, predictor_exclusion_indices)


        # Prepare to add new observation if number exceeds predictions
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



        if len(current_predictors) > len(observations):
            mask = np.in1d(np.arange(len(current_predictors)), prediction_indices)

            unused_indices = np.where(~mask)[0]
            #If assignment not made for a while, increment strikes...threshold is arbitrary 

            for i in unused_indices:
                self.strikes[j] += 1

        # print self.current_predictor_labels

        # Return the label (numerical) of each assignment
        labels = [self.predictors[i]['label'] for i in prediction_indices]
        self.k += 1
        if self.rankings:
            pass # self.cumullog_likelihood(labels)
        if self.show_predictions:
            
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

        rules = {}

        new_index = len(self.predictors)

        # lower_length_limit, upper_length_limit = -600, 600
        # lower_angle_limit, upper_angle_limit =  -4 * np.pi / 9, 4 * np.pi / 9

        # length_diff = ctrl.Antecedent(np.arange(lower_length_limit, upper_length_limit, 1), 'length_diff')
        # angle_diff = ctrl.Antecedent(np.linspace(lower_angle_limit, upper_angle_limit ), 'angle_diff')
        # # foly_diff = ctrl.Antecedent(np.arange(-150, 150), 'foly_diff')

        # congruity = ctrl.Consequent(np.arange(0, 1), 'congruity')

        # length_diff['shorter'] = fuzz.trimf(length_diff.universe, [-600, -600 , -20])
        # length_diff['even'] = fuzz.trimf(length_diff.universe, [-20, 0, 20])
        # length_diff['longer'] = fuzz.trimf(length_diff.universe, [20, 600, 600])

        # angle_diff['more protracted'] = fuzz.trimf(angle_diff.universe, [lower_angle_limit, lower_angle_limit, -np.pi/32])
        # angle_diff['even'] = fuzz.trimf(angle_diff.universe, [-np.pi/32, 0, np.pi/32])
        # angle_diff['more retracted'] = fuzz.trimf(angle_diff.universe, [np.pi/32, upper_angle_limit, upper_angle_limit])

        # congruity['awful'] = fuzz.trimf(congruity.universe, [0, 0, 0.33])
        # congruity['average'] = fuzz.trimf(congruity.universe, [0.33, 0.5, 0.67])
        # congruity['great'] = fuzz.trimf(congruity.universe, [0.67, 1, 1])

        for i, predictor in enumerate(self.predictors):
            tipx, tipy, folx, foly, pixlen = x0[0], x0[1], x0[2], x0[3], x0[4]

            x = predictor['predictor'].x
            tipx_predictor, tipy_predictor, folx_predictor, foly_predictor, pixlen_predictor = x[0], x[1], x[2], x[3], x[4]

            intersection = seg_intersect([folx, foly], [tipx, tipy], [folx_predictor, foly_predictor], [tipx_predictor, tipy_predictor])
            intersection_dist = np.linalg.norm(intersection - np.array([folx, foly]))

            self.predictors[i]['rules'][new_index] = {}

            pixlen_rule = ''
            fol_rule = ''
            closeness_rule = ''
            intersection_rule = ''

            if pixlen - pixlen_predictor < -20:
                pixlen_rule = 'shorter'
            elif pixlen - pixlen_predictor > 20:
                pixlen_rule = 'longer'
            else:
                pixlen_rule = 'even'

            if foly - foly_predictor <= 0:
                fol_rule = 'above'
            elif foly - foly_predictor > 0:
                fol_rule = 'below'


            if np.abs(foly - foly_predictor) < 30 :
                closeness_rule = 'near'
            else:
                closeness_rule = 'far'


            if intersection_dist < 30:
                intersection_rule = 'intersected'
            else:
                intersection_rule = 'not intersected'



            rule_dict = {
                'length_rule': pixlen_rule,
                'fol_rule' : fol_rule, 
                'closeness_rule' : closeness_rule,
                'intersection_rule': intersection_rule
            }
            print "{}->{} \t rules: {}".format(new_index, i, rule_dict) 

            rules[i] = FollicleConstraint(rule_dict)

            opp_rules = {}

            if pixlen_rule == 'shorter':
                opp_rules['length_rule'] = 'longer'
            elif pixlen_rule == 'longer':
                opp_rules['length_rule'] = 'shorter'
            else:
                opp_rules['length_rule'] = 'even'

            if fol_rule == 'above':
                opp_rules['fol_rule'] = 'below'
            elif fol_rule == 'below':
                opp_rules['fol_rule'] = 'above'


            if closeness_rule == 'near':
                opp_rules['closeness_rule'] = 'far'
            else:
                opp_rules['closeness_rule'] = 'near'

            opp_rules['intersection_rule'] = intersection_rule


            self.predictors[i]['rules'][new_index] = FollicleConstraint(opp_rules)


        self.predictors.append(
          {
            'predictor' : kf,
            'rules' : rules,
            'label' : self.current_predictor_labels.pop()
          }
        )


        self.strikes.append(0)
        return new_index

        # self.current_predictor_label += 1

    def cumulative_log_likelihood(self, labels, orders, observation_count):
        rankings = self.rankings
        # observation_count = len(labels)
        distributions = [rankings[observation_count][label] for label in labels]
        # print distributions
        # order = range(len(labels))

        prob = logpmf(orders, distributions, self.max_object_count)
        return prob

    def individual_log_likelihood(self, label, order, observation_count):
        rankings = self.rankings
        # observation_count = len(labels)
        distribution = rankings[observation_count][label]


        return logpmf_single(order, distribution, self.max_object_count)





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





