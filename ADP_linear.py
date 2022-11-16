import random

ME=1
OPPO=2

"""
Features: no. of
* live four
* sleep four
* live three
* sleep three
* live two
* sleep two
of both sides
"""

weights=[0 for _ in range(12)]


class ENV:
    def __init__(self,board,height,width,role):
        self.board=board
        self.height=height
        self.width=width
        self.role=role

    def reset(self,role):
        """
        reset the board
        with role as the offensive
        """

    def is_end(self):
        """
        return True if the game is ended
        """

    def get_actions(self):
        """
        return the possible actions
        """

    def features(self):
        """
        return the features
        """

    def make_move(self,action):
        """
        apply the change according to the action
        return reward
        """

    def draw_move(self,action):
        """
        reverse the effect of last action
        """


def eps_greedy(actions, values, eps, role):
    """
    return the suggested action by Epsilon-greedy method
    """
    if random.random() < eps:
        random_choice = random.randint(0, len(actions) - 1)
        return actions[random_choice], values[random_choice]
    if role == ME:
        best_value = float("-inf")
        for value, action in zip(values, actions):
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = float("inf")
        for value, action in zip(values, actions):
            if value < best_value:
                best_value = value
                best_action = action
    return best_action, best_value


class ADP:
    def __init__(self,discount,learning_rate):
        self.discount=discount
        self.learning_rate=learning_rate

    def value(self,env):
        """
        return the estimated value
        """
        features=env.features()
        value=0
        for i in range(len(features)):
            value+=features[i]*weights[i]


    def update(self,reward,current_value,current_features,next_state_value):
        """
        update weights given a transition
        """
        for i,w in enumerate(weights):
            w+=self.learning_rate*(current_value-reward-self.discount*next_state_value)*current_features[i]

    def train(self,env):
        """
        train to get the fittest weights
        """