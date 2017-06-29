
# coding: utf-8

# # Assignment 2: Markov Decision Processes
# 
# 
# ## Homework Instructions
# All your answers should be written in this notebook.  You shouldn't need to write or modify any other files.
# Look for four instances of "YOUR CODE HERE"--those are the only parts of the code you need to write. To grade your homework, we will check whether the printouts immediately following your code match up with the results we got. The portions used for grading are highlighted in yellow. (However, note that the yellow highlighting does not show up when github renders this file.)
# 
# To submit your homework, send an email to <berkeleydeeprlcourse@gmail.com> with the subject line "Deep RL Assignment 2" and two attachments:
# 1. This `ipynb` file
# 2. A pdf version of this file (To make the pdf, do `File - Print Preview`)
# 
# The homework is due Febrary 22nd, 11:59 pm.
# 
# --------------------------

# ## Introduction
# 
# This assignment will review the two classic methods for solving Markov Decision Processes (MDPs) with finite state and action spaces.
# We will implement value iteration (VI) and policy iteration (PI) for a finite MDP, both of which find the optimal policy in a finite number of iterations.
# 
# The experiments here will use the Frozen Lake environment, a simple gridworld MDP that is taken from `gym` and slightly modified for this assignment. In this MDP, the agent must navigate from the start state to the goal state on a 4x4 grid, with stochastic transitions.

# In[2]:

from frozen_lake import FrozenLakeEnv
env = FrozenLakeEnv()
print(env.__doc__)


# Let's look at what a random episode looks like.

# In[2]:

# Some basic imports and setup
import numpy as np, numpy.random as nr, gym
np.set_printoptions(precision=3)
def begin_grading(): print("\x1b[43m")
def end_grading(): print("\x1b[0m")

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done
env.render();


# In the episode above, the agent falls into a hole after two timesteps. Also note the stochasticity--on the first step, the DOWN action is selected, but the agent moves to the right.
# 
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

# In[3]:

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)


print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, which transitions to itself with probability 1 and reward 0.")
print("P[5][0] =", mdp.P[5][0], '\n')


# ## Part 1: Value Iteration

# ### Problem 1: implement value iteration
# In this problem, you'll implement value iteration, which has the following pseudocode:
# 
# ---
# Initialize $V^{(0)}(s)=0$, for all $s$
# 
# For $i=0, 1, 2, \dots$
# - $V^{(i+1)}(s) = \max_a \sum_{s'} P(s,a,s') [ R(s,a,s') + \gamma V^{(i)}(s')]$, for all $s$
# 
# ---
# 
# We additionally define the sequence of greedy policies $\pi^{(0)}, \pi^{(1)}, \dots, \pi^{(n-1)}$, where
# $$\pi^{(i)}(s) = \arg \max_a \sum_{s'} P(s,a,s') [ R(s,a,s') + \gamma V^{(i)}(s')]$$
# 
# Your code will return two lists: $[V^{(0)}, V^{(1)}, \dots, V^{(n)}]$ and $[\pi^{(0)}, \pi^{(1)}, \dots, \pi^{(n-1)}]$
# 
# To ensure that you get the same policies as the reference solution, choose the lower-index action to break ties in $\arg \max_a$. This is done automatically by np.argmax. This will only affect the "# chg actions" printout below--it won't affect the values computed.
# 
# <div class="alert alert-warning">
# Warning: make a copy of your value function each iteration and use that copy for the update--don't update your value function in place. 
# Updating in-place is also a valid algorithm, sometimes called Gauss-Seidel value iteration or asynchronous value iteration, but it will cause you to get different results than me.
# </div>

# In[4]:

def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)
        
    len(value_functions) == nIt+1 and len(policies) == n
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1] # V^{(it)}
        # YOUR CODE HERE
        # Your code should define the following two variables
        # pi: greedy policy for Vprev, 
        #     corresponding to the math above: \pi^{(it)} = Greedy[V^{(it)}]
        #     numpy array of ints
        # V: bellman backup on Vprev
        #     corresponding to the math above: V^{(it+1)} = T[V^{(it)}]
        #     numpy array of floats

        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)

        for s in range(mdp.nS):
            v = []
            # compute the expectation of the reward for each action
            for a in range(mdp.nA):
                mean_reward = 0.
                for prob, next_s, reward in mdp.P[s][a]:
                    mean_reward += prob * (reward + gamma * Vprev[next_s])
                v.append(mean_reward)

            V[s] = np.max(v)
            pi[s] = np.argmax(v)
                    
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

GAMMA=0.95 # we'll be using this same value in subsequent problems
begin_grading()
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
end_grading()


# Below, we've illustrated the progress of value iteration. Your optimal actions are shown by arrows.
# At the bottom, the value of the different states are plotted.

# In[5]:

import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
for (V, pi) in zip(Vs_VI[:10], pis_VI[:10]):
    plt.figure(figsize=(3,3))
    plt.imshow(V.reshape(4,4), cmap='gray', interpolation='none', clim=(0,1))
    ax = plt.gca()
    ax.set_xticks(np.arange(4)-.5)
    ax.set_yticks(np.arange(4)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    Y, X = np.mgrid[0:4, 0:4]
    a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
    Pi = pi.reshape(4,4)
    for y in range(4):
        for x in range(4):
            a = Pi[y, x]
            u, v = a2uv[a]
            plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1) 
            plt.text(x, y, str(env.desc[y,x].item().decode()),
                     color='g', size=12,  verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
    plt.grid(color='b', lw=2, ls='-')
plt.figure()
plt.plot(Vs_VI)
plt.title("Values of different states");
#plt.show()

print("[ end of problem 1. ]")

# ## Problem 2: construct an MDP where value iteration takes a long time to converge
# 
# When we ran value iteration on the frozen lake problem, the last iteration where an action changed was iteration 6--i.e., value iteration computed the optimal policy at iteration 6.
# Are there any guarantees regarding how many iterations it'll take value iteration to compute the optimal policy?
# There are no such guarantees without additional assumptions--we can construct the MDP in such a way that the greedy policy will change after arbitrarily many iterations.
# 
# Your task: define an MDP with at most 3 states and 2 actions, such that when you run value iteration, the optimal action changes at iteration >= 50. Use discount=0.95. (However, note that the discount doesn't matter here--you can construct an appropriate MDP with any discount.)

# In[6]:

chg_iter = 50
# YOUR CODE HERE
# Your code will need to define an MDP (mymdp)
# like the frozen lake MDP defined above
nS = 2
nA = 2
P = {}
P[0] = {0: [(0.9, 0, 0), (0.1, 1, 0)], 
        1: [(1.0, 0, 1)]}
P[1] = {0: [(0.1, 0, 0), (0.9, 1, 0)], 
        1: [(1.0, 1, 1.571)]}
mymdp = MDP(P, nS, nA)

begin_grading()
Vs, pis = value_iteration(mymdp, gamma=GAMMA, nIt=chg_iter+1)
end_grading()

print("[ end of problem 2. ]")

# ## Problem 3: Policy Iteration
# 
# The next task is to implement exact policy iteration (PI), which has the following pseudocode:
# 
# ---
# Initialize $\pi_0$
# 
# For $n=0, 1, 2, \dots$
# - Compute the state-value function $V^{\pi_{n}}$
# - Using $V^{\pi_{n}}$, compute the state-action-value function $Q^{\pi_{n}}$
# - Compute new policy $\pi_{n+1}(s) = \operatorname*{argmax}_a Q^{\pi_{n}}(s,a)$
# ---
# 
# Below, you'll implement the first and second steps of the loop.
# 
# ### Problem 3a: state value function
# 
# You'll write a function called `compute_vpi` that computes the state-value function $V^{\pi}$ for an arbitrary policy $\pi$.
# Recall that $V^{\pi}$ satisfies the following linear equation:
# $$V^{\pi}(s) = \sum_{s'} P(s,\pi(s),s')[ R(s,\pi(s),s') + \gamma V^{\pi}(s')]$$
# You'll have to solve a linear system in your code. (Find an exact solution, e.g., with `np.linalg.solve`.)

# In[7]:

def compute_vpi(pi, mdp, gamma):
    # YOUR CODE HERE
    delta = 1
    epsilon = 0.00001
    V = np.zeros(len(pi))

    while delta > epsilon:
        delta = 0
        for s in range(mdp.nS):
            Vcur = 0
            for prob, next_state, reward in mdp.P[s][pi[s]]:
                Vcur += prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s] - Vcur))
            V[s] = Vcur

    return V


# Now let's compute the value of an arbitrarily-chosen policy. 
# 

# In[8]:

begin_grading()
print(compute_vpi(np.ones(16, dtype=int), mdp, gamma=GAMMA))
end_grading()


# As a sanity check, if we run `compute_vpi` on the solution from our previous value iteration run, we should get approximately (but not exactly) the same values produced by value iteration.

# In[9]:

Vpi=compute_vpi(pis_VI[15], mdp, gamma=GAMMA)
V_vi = Vs_VI[15]
print("From compute_vpi", Vpi)
print("From value iteration", V_vi)
print("Difference", Vpi - V_vi)

print("[ end of problem 3a. ]")

# ### Problem 3b: state-action value function
# 
# Next, you'll write a function to compute the state-action value function $Q^{\pi}$, defined as follows
# 
# $$Q^{\pi}(s, a) = \sum_{s'} P(s,a,s')[ R(s,a,s') + \gamma V^{\pi}(s')]$$
# 

# In[10]:

def compute_qpi(vpi, mdp,  gamma):
    # YOUR CODE HERE
    Qpi = np.zeros([mdp.nS, mdp.nA])
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for prob, next_state, reward in mdp.P[s][a]:
                Qpi[s][a] += prob * (reward + gamma * vpi[next_state])
    return Qpi

begin_grading()
Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Qpi:\n", Qpi)
end_grading()


# Now we're ready to run policy iteration!

# In[11]:

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis
Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);


# Now we can compare the convergence of value iteration and policy iteration on several states.
# For fun, you can try adding modified policy iteration.

# In[12]:

for s in range(5):
    plt.figure()
    plt.plot(np.array(Vs_VI)[:,s])
    plt.plot(np.array(Vs_PI)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')

print("[ end of problem 3b. ]")

# In[ ]:



