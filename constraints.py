import abc
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Constraint(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        congruity = ctrl.Consequent(np.linspace(0, 1), 'congruity')

        congruity['awful'] = fuzz.trimf(congruity.universe, [0, 0, 0.33])
        congruity['average'] = fuzz.trimf(congruity.universe, [0.33, 0.5, 0.67 ])
        congruity['great'] = fuzz.trimf(congruity.universe, [0.67, 0.67, 1])

        self.congruity = congruity        

    def compute_congruity(self, comp_dict):
        congruity_control = self.congruity_control
        congruity_calculator = ctrl.ControlSystemSimulation(congruity_control)

        congruity_calculator.inputs(comp_dict)
        congruity_calculator.compute()

        return congruity_calculator.output['congruity']

class AngleConstraint(Constraint):
    def __init__(self, rule_dict):
        # rule_dict = {
        #     'length_rule': shorter,
        #     'angle_rule': longer,
        #     'fol_distance': more protracted,
        # }

        super(AngleConstraint, self).__init__()

        lower_length_limit, upper_length_limit = -600, 600
        lower_angle_limit, upper_angle_limit =  -4 * np.pi / 9, 4 * np.pi / 9

        length_diff = ctrl.Antecedent(np.arange(lower_length_limit, upper_length_limit, 1), 'length_diff')
        angle_diff = ctrl.Antecedent(np.linspace(lower_angle_limit, upper_angle_limit), 'angle_diff')
        # foly_diff = ctrl.Antecedent(np.arange(-150, 150), 'foly_diff')

        congruity = self.congruity

        # length_diff['shorter'] = fuzz.trimf(length_diff.universe, [-600, -600 , -20])
        # length_diff['even'] = fuzz.trimf(length_diff.universe, [-20, 0, 20])
        # length_diff['longer'] = fuzz.trimf(length_diff.universe, [20, 600, 600])

        length_diff['shorter'] = fuzz.trapmf(length_diff.universe, [-600, -600, -40, -20])
        length_diff['even'] = fuzz.trimf(length_diff.universe, [-20, 0, 20])
        length_diff['longer'] = fuzz.trapmf(length_diff.universe, [20, 40, 600, 600])

        # angle_diff['more retracted'] = fuzz.trimf(angle_diff.universe, [lower_angle_limit, lower_angle_limit, -np.pi/32])
        # angle_diff['even'] = fuzz.trimf(angle_diff.universe, [-np.pi/32, 0, np.pi/32])
        # angle_diff['more protracted'] = fuzz.trimf(angle_diff.universe, [np.pi/32, upper_angle_limit, upper_angle_limit])

        angle_diff['more retracted'] = fuzz.trapmf(angle_diff.universe, [lower_angle_limit, lower_angle_limit, -np.pi/16, -np.pi/32])
        angle_diff['even'] = fuzz.trimf(angle_diff.universe, [-np.pi/32, 0, np.pi/32])
        angle_diff['more protracted'] = fuzz.trapmf(angle_diff.universe, [np.pi/32, np.pi/16, upper_angle_limit, upper_angle_limit])


        rule1 = ctrl.Rule(
            length_diff[rule_dict['length_rule']] &
            angle_diff[rule_dict['angle_rule']],
            congruity['great']
        )

        # rule2 = ctrl.Rule(
        #     length_diff[rule_dict['length_rule']] |
        #     angle_diff[rule_dict['angle_rule']],
        #     congruity['average']
        # )

        # rule3 = ctrl.Rule(
        #     ~length_diff[rule_dict['length_rule']] &
        #     ~angle_diff[rule_dict['angle_rule']],
        #     congruity['awful']
        # )

        rule2 = ctrl.Rule(
            ~angle_diff[rule_dict['angle_rule']],
            congruity['awful']
        )

        rule3 = ctrl.Rule(
            ~length_diff[rule_dict['length_rule']] &
            angle_diff[rule_dict['angle_rule']],
            congruity['awful']
        )

        self.congruity_control = ctrl.ControlSystem([rule1, rule2, rule3])
        self.rule_dict = rule_dict

class FollicleConstraint(Constraint):
    def __init__(self, rule_dict):
        super(FollicleConstraint, self).__init__()
        congruity = self.congruity

        self.rule_dict = rule_dict

        lower_length_limit, upper_length_limit = -600, 600

        length_diff = ctrl.Antecedent(np.arange(lower_length_limit, upper_length_limit, 1), 'length_diff')
        length_diff['shorter'] = fuzz.trapmf(length_diff.universe, [-600, -600, -40, 0])
        length_diff['even'] = fuzz.trimf(length_diff.universe, [-40, 0, 40])
        length_diff['longer'] = fuzz.trapmf(length_diff.universe, [0, 40, 600, 600])

        lower_fol_limit, upper_fol_limit = -300, 300

        fol_diff = ctrl.Antecedent(np.linspace(lower_fol_limit, upper_fol_limit, 1000), 'fol_diff')
        # fol_diff['above'] = fuzz.trapmf(fol_diff.universe, [lower_fol_limit, lower_fol_limit, -5, 0])
        # fol_diff['even'] = fuzz.trimf(fol_diff.universe, [-5, 0, 5])
        # fol_diff['below'] = fuzz.trapmf(fol_diff.universe, [0, 5, upper_fol_limit, upper_fol_limit])

        fol_diff['above'] = fuzz.trapmf(fol_diff.universe, [lower_fol_limit, lower_fol_limit, -5, 5])
        fol_diff['below'] = fuzz.trapmf(fol_diff.universe, [-5, 5, upper_fol_limit, upper_fol_limit])

        overlap = ctrl.Antecedent(np.arange(0, 600), 'overlap')
        # intersection_dist['intersected'] = fuzz.trapmf(intersection_dist.universe, [0, 0, 10, 30])
        # intersection_dist['not intersected'] = fuzz.trapmf(intersection_dist.universe, [20, 40, 300, 300])

        overlap['true'] = fuzz.trapmf(overlap.universe, [0, 0, 5, 10])
        overlap['false'] = fuzz.trapmf(overlap.universe, [7, 12, 600, 600])

        # closeness = ctrl.Antecedent(np.linspace(0, upper_fol_limit, 1000), 'closeness')
        # closeness['near'] = fuzz.trapmf(closeness.universe, [0, 0, 20, 40])
        # closeness['far'] = fuzz.trapmf(closeness.universe, [20, 40, upper_fol_limit, upper_fol_limit])

        length_rule, fol_rule, closeness_rule, overlap_rule = rule_dict['length_rule'], rule_dict['fol_rule'], rule_dict['closeness_rule'], rule_dict['overlap_rule']


        # rule1 = ctrl.Rule(
        #     fol_diff[fol_rule] &
        #     closeness[closeness_rule],
        #     congruity['great']
        # )

        # rule2 = ctrl.Rule(
        #     closeness[closeness_rule] |
        #     (fol_diff[fol_rule] & length_diff[length_rule]),
        #     congruity['great']
        # )

        # rule3 = ctrl.Rule(
        #     ~fol_diff[fol_rule] |
        #     ~length_diff[length_rule],
        #     congruity['great']
        # )

        # rule1 = ctrl.Rule(
        #     length_diff[length_rule] &
        #     fol_diff[fol_rule],
        #     congruity['great']
        # )

        # rule2 = ctrl.Rule(
        #     length_diff[length_rule] |
        #     (fol_diff[fol_rule] & closeness[closeness_rule]),
        #     congruity['average']
        # )

        # rule3 = ctrl.Rule(
        #     ~fol_diff[fol_rule] |
        #     ~closeness[closeness_rule],
        #     congruity['awful']
        # )

        rule1 = ctrl.Rule(
            ~overlap[overlap_rule],
            congruity['awful']
        )

        rule2 = ctrl.Rule(
            fol_diff[fol_rule],
            congruity['great']
        )
        rule3 = ctrl.Rule(
            ~fol_diff[fol_rule],
            congruity['average']
        )






        self.congruity_control = ctrl.ControlSystem([rule1, rule2, rule3])


