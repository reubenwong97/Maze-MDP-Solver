import numpy as np
from learner.learner import Learner

class PolicyIterationLearner(Learner):
    def __init__(self, env):
        super(PolicyIterationLearner, self).__init__(env)
        self.name = "Policy Iteration"