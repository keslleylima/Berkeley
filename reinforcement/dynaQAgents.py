# dynaAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# Dyna Agent support by Anderson Tavares (artavares@inf.ufrgs.br)
import self as self
import state as state
from game import *
from learningAgents import ReinforcementAgent

import random,util,math

class DynaQAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
        - self.plan_steps (number of planning iterations)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, plan_steps=5, kappa=0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.plan_steps = plan_steps
        self.kappa = kappa
        self.q_table = dict()
        self.model = dict()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_table[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) > 0:
            return max([self.getQValue(state, action) for action in legalActions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) > 0:
            return max(legalActions, key=lambda action: self.getQValue(state, action))
        else:
            return None
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)

        if len(legalActions) > 0:
            return self.computeActionFromQValues(state)
        elif util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return None

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here.

          NOTE: You should never call this function,
          it will be called on your behalf

          NOTE2: insert your planning code here as well
        """
        self.q_table[(state,action)] = self.getQValue(state, action) + self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action))

        if state not in self.model:
            self.model[state] = {act: (0, state, 0) for act in self.getLegalActions(state)}

        self.model[state][action] = (reward, nextState, 0)
        
        for _ in range(self.plan_steps):
            self.q_table[(random.choice(self.model.keys()),random.choice(self.model[random.choice(
                self.model.keys())].keys()))] = self.getQValue(random.choice(self.model.keys()),
                                                               random.choice(self.model[random.choice(self.model.keys())].keys()))\
                                                + self.alpha * (self.model[random.choice(
                self.model.keys())][random.choice(self.model[random.choice(self.model.keys())].keys())]
                                                                + self.kappa*math.sqrt(
                        self.model[random.choice(self.model.keys())][random.choice(self.model[random.choice(self.model.keys())].keys())])
                                                                + self.discount*self.computeValueFromQValues(self.model[random.choice(
                        self.model.keys())][random.choice(self.model[random.choice(self.model.keys())].keys())]) -
                                                                self.getQValue(random.choice(self.model.keys()),
                                                                               random.choice(self.model[random.choice(self.model.keys())].keys())))
            act: object
            for act in self.model[random.choice(self.model.keys())].keys():
              if random.choice(self.model[random.choice(self.model.keys())].keys()) != act:
                self.model[random.choice(self.model.keys())][act] = (self.model[random.choice(self.model.keys())][act][0], self.model[random.choice(self.model.keys())][act][1], self.model[random.choice(self.model.keys())][act][2] + 1)
              else:
                self.model[random.choice(self.model.keys())][act] = (self.model[random.choice(self.model.keys())][act][0], self.model[random.choice(self.model.keys())][act][1], 0)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanDynaQAgent(DynaQAgent):
    "Exactly the same as DynaAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanDynaAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        DynaQAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of DynaAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = DynaQAgent.getAction(self, state)
        self.doAction(state,action)
        return action