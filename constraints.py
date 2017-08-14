# Constraints based around scikit-fuzzy fuzzy logic
# Explore http://pythonhosted.org/scikit-fuzzy/ for information and tutorials
# Also look at https://link.springer.com/content/pdf/10.1007%2F978-3-642-17289-2_38.pdf for example of implementation in object tracking

import abc
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

#Define common antecedents and consequents.
#For example check out http://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html

LOWER_LENGTH_LIMIT, UPPER_LENGTH_LIMIT = -600, 600

LENGTH_DIFF = ctrl.Antecedent(np.arange(LOWER_LENGTH_LIMIT, UPPER_LENGTH_LIMIT, 1), 'length_diff')
LENGTH_DIFF['shorter'] = fuzz.trapmf(LENGTH_DIFF.universe, [LOWER_LENGTH_LIMIT, LOWER_LENGTH_LIMIT, -40, 0])
LENGTH_DIFF['longer'] = fuzz.trapmf(LENGTH_DIFF.universe, [0, 40, UPPER_LENGTH_LIMIT, UPPER_LENGTH_LIMIT])

LOWER_AREA_DIFF_LIMIT, UPPER_AREA_DIFF_LIMIT = - (300 ** 2), 300 ** 2

AREA_DIFF = ctrl.Antecedent(np.linspace(LOWER_AREA_DIFF_LIMIT, UPPER_AREA_DIFF_LIMIT, 1000), 'area_diff')
AREA_DIFF['above'] = fuzz.trapmf(AREA_DIFF.universe, [LOWER_AREA_DIFF_LIMIT, LOWER_AREA_DIFF_LIMIT, -2000, 2000])
AREA_DIFF['below'] = fuzz.trapmf(AREA_DIFF.universe, [-2000, 2000, UPPER_AREA_DIFF_LIMIT, UPPER_AREA_DIFF_LIMIT])

OVERLAP = ctrl.Antecedent(np.arange(0, 600), 'overlap')
OVERLAP['true'] = fuzz.trapmf(OVERLAP.universe, [0, 0, 5, 10])
OVERLAP['false'] = fuzz.trapmf(OVERLAP.universe, [7, 12, 600, 600])

CONGRUITY = ctrl.Consequent(np.linspace(0, 1), 'congruity')

CONGRUITY['awful'] = fuzz.trimf(CONGRUITY.universe, [0, 0, 0.5])
CONGRUITY['average'] = fuzz.trimf(CONGRUITY.universe, [0, 0.5, 1])
CONGRUITY['great'] = fuzz.trimf(CONGRUITY.universe, [0.5, 1, 1])



class Constraint(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """ Base class for constraints based around fuzzy logic
            End result is to return probability that a hypothesis is true with the in place constraints


        """
        congruity = ctrl.Consequent(np.linspace(0, 1), 'congruity')

        # Define 3 tiers of congruity as 3 probablity ranges
        # congruity['awful'] = fuzz.trimf(congruity.universe, [0, 0, 0.33])
        # congruity['average'] = fuzz.trimf(congruity.universe, [0.33, 0.5, 0.67 ])
        # congruity['great'] = fuzz.trimf(congruity.universe, [0.67, 1, 1])

        congruity['awful'] = fuzz.trimf(congruity.universe, [0, 0, 0.5])
        congruity['average'] = fuzz.trimf(congruity.universe, [0, 0.5, 1])
        congruity['great'] = fuzz.trimf(congruity.universe, [0.5, 1, 1])

        self.congruity = congruity        

    def compute_congruity(self, value_dict):
        """ Calculate congruity based on comparisons

            value_dict: Dictionary containing keys and values expected 
                        by the constraint class

        """
        congruity_control = self.congruity_control
        congruity_calculator = ctrl.ControlSystemSimulation(congruity_control)

        congruity_calculator.inputs(value_dict)
        congruity_calculator.compute()
        # print "rule: {}\t closeness: {}\t congruity: {}".format(self.rule_dict['closeness_rule'], value_dict['closeness'], congruity_calculator.output['congruity'])
        return congruity_calculator.output['congruity']


class FollicleConstraint(Constraint):
    def __init__(self, rule_dict):
        """ Constraint that puts heavy emphasis on relationships between
            whisker follicles

            rule_dict : dictionary of rules in english to pass to constraint

            (todo: consider passing rules as values instead of words)

        """

        # rule_dict = {
        #     length_rule : (50, 5),

        # }

        super(FollicleConstraint, self).__init__()

        congruity = self.congruity
        self.rule_dict = rule_dict
        length_rule, fol_rule, closeness_rule, overlap_rule = rule_dict['length_rule'], rule_dict['fol_rule'], rule_dict['closeness_rule'], rule_dict['overlap_rule']


        closeness = ctrl.Antecedent(np.linspace(-1, 1), 'closeness')
        closeness['good'] = fuzz.trimf(closeness.universe, [closeness_rule - 0.9, closeness_rule, closeness_rule + 0.9])
        # closeness['far'] = fuzz.trimf(closeness.universe, [3, 10, 10])


        length_diff = ctrl.Antecedent(np.linspace(-600, 600), 'length_diff')
        length_diff['good'] = fuzz.trimf(length_diff.universe, [length_rule - 200, length_rule, length_rule + 200])


        # Define rules that will place congruity into the 3 tiers
        # Is a good way of defining which constraints are more important

        rule1 = ctrl.Rule(
            ~closeness['good'],
            congruity['awful']
        )

        rule2 = ctrl.Rule(
            length_diff['good'],
            congruity['great']
        )

        rule3 = ctrl.Rule(
            ~length_diff['good'],
            congruity['average']
        )

        self.congruity_control = ctrl.ControlSystem([rule1, rule2, rule3])

class LengthFocusedConstraint(Constraint):
    def __init__(self):
        """
            Constraint with emphasis on length differences

        """
        super(LengthFocusedConstraint, self).__init__()


    def set_rules(self, rule_dict):
        length_rule, area_rule = rule_dict['length_rule'], rule_dict['area_rule']


        rule1 = ctrl.Rule(
            ~LENGTH_DIFF[length_rule],
            CONGRUITY['awful'],
        )

        rule2 = ctrl.Rule(
            AREA_DIFF[area_rule],
            CONGRUITY['great'],
        )

        rule3 = ctrl.Rule(
            ~AREA_DIFF[area_rule],
            CONGRUITY['average'],
        )

        rules = [rule1, rule2, rule3]

        self.rules = rules
        self.congruity_control = ctrl.ControlSystem(rules)

class HeightFocusedConstraint(Constraint):

    def __init__(self):
        """
            Constraint with emphasis on height differences
        """
        super(HeightFocusedConstraint, self).__init__()


    def set_rules(self, rule_dict):
        length_rule, area_rule = rule_dict['length_rule'], rule_dict['area_rule']


        rule1 = ctrl.Rule(
            ~AREA_DIFF[area_rule],
            CONGRUITY['awful'],
        )

        rule2 = ctrl.Rule(
            LENGTH_DIFF[length_rule],
            CONGRUITY['great'],
        )

        rule3 = ctrl.Rule(
            ~LENGTH_DIFF[length_rule],
            CONGRUITY['average'],
        )

        rules = [rule1, rule2, rule3]

        self.rules = rules
        self.congruity_control = ctrl.ControlSystem(rules)


