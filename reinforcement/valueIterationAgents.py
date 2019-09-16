# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()

        states = mdp.getStates()

        if len(states) > 0:
            for i in range(0, self.iterations):
                next_values = util.Counter()
                for state in states:
                    values = []
                    possible_actions = mdp.getPossibleActions(state)
                    if self.mdp.isTerminal(state):
                        values.append(0)
                    for action in possible_actions:
                        if action is not None:
                            values.append(self.computeQValueFromValues(state, action))

                    next_values[state] = max(values)

                self.values = next_values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          Q*(s,a) = sum[s'] T(s,a,s')[R(s,a,s')+a.V_{k}(s)]
        """
        qValue = 0
        transStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        for transStatesAndProb in transStatesAndProbs:
            qValue += transStatesAndProb[1] * (self.mdp.getReward(state, action, transStatesAndProb[0]) + self.discount
                                               * self.getValue(transStatesAndProb[0]))
        return qValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          V_{k+1}(s) = max[a] Q*(s,a)
        """
        # Possibleactions get the list of possible actions from the state as a parameter
        possibleActions = self.mdp.getPossibleActions(state)

        # Compare if the state is a terminal state or possibleActions is empty
        if self.mdp.isTerminal(state) or len(possibleActions) <= 0:
            return None
        else:
            value = self.getQValue(state, possibleActions[0])
            action = possibleActions[0]

            for possibleAction in possibleActions:
                auxValue = self.getQValue(state, possibleAction)
                if value is not None and auxValue is not None:
                    if value <= auxValue:
                        value = auxValue
                        action = possibleAction

            return action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
