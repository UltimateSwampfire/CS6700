import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3

class GridWorldEnv(object):
    '''
        start_states: list of coordinates for starting state
        goal_states: list of coordinates for goal state
        max_steps: maximum steps before episode terminates
        action_fail_prob: probability with which chosen action fails and a random action is instead taken
    '''
    def __init__(self, grid_file, start_states = [(0,0)], goal_states = [(10,10)], goal_reward = 10, max_steps = 100,
                    action_fail_prob = 0.0, seed = None):
        # Load the grid from txt file
        self.grid = np.loadtxt(grid_file, delimiter=' ').astype('int')
        self.random_generator = np.random.RandomState(seed)

        self.start_states = start_states
        self.goal_states = goal_states
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.action_fail_prob = action_fail_prob
        self.action_space = [UP, DOWN, LEFT, RIGHT]
        self.reset()
    
    '''
    is_in_grid: Check to see if given coord_x,coord_y coordinates is inside the grid
    '''
    def is_in_grid(self, coord_x, coord_y):
        if coord_x < 0 or coord_y<0:
            return False
        if coord_x >= self.grid.shape[0] or coord_y >= self.grid.shape[1]:
            return False
        
        return True
    '''
    choose_state: choose a state from the list
    list_state: list of tupules of two integers

    '''
    def choose_state(self, list_states):
        choice = self.random_generator.randint(len(list_states))
        return list_states[choice]


    '''
    reset: resets the environment by randomly choosing a start and goal state
    '''
    def reset(self, start_state=None, goal_state=None):
        self.start_state = self.state = self.choose_state(self.start_states) if start_state is None else start_state
        #self.goal = self.choose_state(self.goal_states) if goal_state is None else goal_state
        self.done = False
        self.steps = 0
        return self.state

    '''
    step: change the state after taking action
    Returns: new state, environment reward, done (whether episode is completed)
    '''
    def step(self, action):
        assert action in self.action_space, "Wrong action %d chosen, Possible actions: %s"%(action, str(self.action_space))
        if self.done:
            print("Warning: Episode done")
        
        self.steps += 1

        # With prob = self.action_fail_prob choose a random action
        if self.random_generator.rand() < self.action_fail_prob:
            action = self.action_space[self.random_generator.randint(len(self.action_space))]

        if action == UP:
            new_state = (self.state[0]+1, self.state[1])
        elif action == DOWN:
            new_state = (self.state[0]-1, self.state[1])
        elif action == LEFT:
            new_state = (self.state[0], self.state[1]-1)
        elif action == RIGHT:
            new_state = (self.state[0], self.state[1]+1)
        
        if self.is_in_grid(new_state[0], new_state[1]):
            self.state = new_state
        
        if self.state in self.goal_states:
            self.reward = self.goal_reward
            self.done = True
            return self.state, self.reward, self.done
        
        if self.steps >= self.max_steps:
            self.done = True
        
        self.reward = - self.grid[self.state[0], self.state[1]]

        return self.state, self.reward, self.done

    '''
    render: render a plot of the environment
    '''
    def render(self, render_agent = False, ax = None):
        grid = self.grid.copy()
        for start in self.start_states:
            grid[start[0], start[1]] = 3
        for goal in self.goal_states:
            grid[goal[0], goal[1]] = 4
        if render_agent:
            grid[self.state[0], self.state[1]] = 5
        
        plt.clf()
        if not render_agent:
            cmap = colors.ListedColormap(['#F5E5E1', '#F2A494', '#FF2D00', '#0004FF', '#00FF23'])
        else:
            cmap = colors.ListedColormap(['#F5E5E1', '#F2A494', '#FF2D00', '#0004FF', '#00FF23', '#F0FF00'])
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.pcolor(grid, cmap=cmap, edgecolors='k', linewidths=2)
        
    
    '''
    render_policy: render a learnt policy
    '''
    def render_policy(self, policy):
        pass

'''
Grid World environment with leftward wind: extra left action with 0.5 probability
'''
class GridWorldWindyEnv(GridWorldEnv):
    def __init__(self, grid_file, start_states = [(0,0)], goal_states = [(10,10)], goal_reward = 10, max_steps = 100,
                    action_fail_prob = 0.0, seed = None, windy_probab = 0.5):
        super(GridWorldWindyEnv, self).__init__(grid_file, start_states, goal_states , goal_reward , max_steps,
                    action_fail_prob , seed)
        self.windy_probab = windy_probab
    
    def step(self, action):
        ans = super().step(action)
        if not self.done and self.random_generator.rand() < self.windy_probab:
            ans1 = super().step(LEFT)
            return ans1[0], ans[1]+ans1[1], ans1[2]
        return ans

def plot_Q(Q, message = "Q plot"):
    
    plt.figure(figsize=(10,10))
    plt.title(message)
    plt.pcolor(Q.max(-1), edgecolors='k', linewidths=2)
    plt.colorbar()
    def x_direct(a):
        if a in [UP, DOWN]:
            return 0
        return 1 if a == RIGHT else -1
    def y_direct(a):
        if a in [RIGHT, LEFT]:
            return 0
        return 1 if a == UP else -1
    policy = Q.argmax(-1)
    policyx = np.vectorize(x_direct)(policy)
    policyy = np.vectorize(y_direct)(policy)
    idx = np.indices(policy.shape)
    plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
    plt.show()