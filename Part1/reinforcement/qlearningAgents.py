# qlearningAgents.py
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
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

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        print "I am inside __init__"
        print args
        "*** YOUR CODE HERE (Done) ***"
        '''
        -> Initializing epsilon, discount, alpha values as
        this->alpha = alpha
        this->discount = gamma
        this->epsilon = epsilon
        
        -> Setting the actionFn passed from environment to be used in legalActions from any state
        this->actionFn = actionFn
        
        -> Initializing the qValues to 0.0
        this->qValues => 0.0
        '''

        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE (Done)  ***"
        '''
        -> Get Q_t(state, action) at any point in time t
        return this->qValues(state, action)
        '''

        return self.qValues[(state, action)]
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE (Done)  ***"
        '''
        -> Get the V_t value of any state 's' at any point in time 't'
        by getting the max of Q_t(s, a_i) for all legalActions 'a_i' in state 's'
        
        V = 0.0
        for action in AllLegalActions in state 's'
            => if V < Q(s, action)
                    V = Q(s, action)
                    
        return V
        
        '''

        max_action_value = 0.0
        for act in self.getLegalActions(state):
            if max_action_value < self.getQValue(state, act):
                max_action_value = self.getQValue(state, act)

        return max_action_value
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE (Done)  ***"
        '''
        -> Get the most perceived rewarding action which corresponds to V_t at any state 's'
        by storing the action while calculating the V_t ie max of Q_t(s, a_i)
        
        V = 0.0, Action_max = ''
        for action in AllLegalActions in state 's'
            => if Action_max is ''
                Action_max = action
                
            => if V < Q(s, action)
                    V = Q(s, action)
                    Action_max = action
                    
        return Action_max
        
        '''

        max_action_value, max_action = 0.0, None
        for act in self.getLegalActions(state):
            if max_action is None:
                max_action = act

            if max_action_value < self.getQValue(state, act):
                max_action_value = self.getQValue(state, act)
                max_action = act

        #print "I am inside computeActionFromQValues"

        return max_action


        #util.raiseNotDefined()

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
        action = None
        "*** YOUR CODE HERE (Done)  ***"
        '''
        -> Get random action with probability defined as epsilon 
        but otherwise take the action as per the policy
        Move to 'west' in terminal state to continue the iterations
        
        Actions(s) => All legal actions in state 's'
        => if Actions(s) is empty
                move to 'west'
            
           Otherwise
                r => generate random value between 0 & 1
                if r < epsilom
                    take random action from Actions(s)
                Otherwise
                    take action as per the policy ie Action_max
                    
        '''

        if not legalActions:
            action = 'west'
        elif util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE (Done)  ***"
        '''
        Update the Q_t(s, a) value using the below formula iteratively
        alpha => Learning Rate
        gamma => Discount
        
        Q_t(s, a) => Q_t-1(s, a) + alpha * TD(a, s)
        Q_t(s, a) => Q_t-1(s, a) + alpha * ( Reward(s, a) + gamma * V(next_state) - Q_t-1(s, a) ) 
        Q_t(s, a) => (1 - alpha) * Q_t-1(s, a) + alpha * ( Reward(s, a) + gamma * V(next_state) )
        
        
        '''

        nextStateValue = self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha)*self.qValues[(state, action)] + self.alpha*(reward + self.discount * nextStateValue)
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

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
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        print "I am inside ApproximateQAgent->getQValue"

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        print "I am inside ApproximateQAgent->update"

        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print "I am inside ApproximateQAgent->final"

            pass
