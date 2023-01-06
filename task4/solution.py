import numpy as np
import matplotlib.pyplot as plt

import time
import gym
import scipy.signal
from gym.spaces import Box, Discrete

import joblib
import copy
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
import random
import os
from torch.autograd import Variable
#import tensorflow as tf

"""os.environ['PYTHONHASHSEED'] = 'no randomness'
torch.manual_seed(9876) # avoid randomness
np.random.seed(8098127)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)"""

torch.manual_seed(0) # avoid randomness
np.random.seed(0)
random.seed(0)

USE_NAIVE = False 
PRUNE = False # Whether to return as soon as score is over PRUNE_THRESHOLD
PRUNE_THRESHOLD = 170


def discount_cumsum(x, discount):
    """
    Compute  cumulative sums of vectors.

    Input: [x0, x1, ..., xn]
    Output: [x0 + discount * x1 + discount^2 * x2 + ... , x1 + discount * x2 + ... , ... , xn]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def combined_shape(length, shape=None):
    """Helper function that combines two array shapes."""
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    """The basic multilayer perceptron architecture used."""
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class MLPCategoricalActor(nn.Module):
    """A class for the policy network."""
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """Takes the observation and outputs a distribution over actions."""
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        """
        Takes a distribution and action, then gives the log-probability of the action
        under that distribution.
        """
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        """
        Produce action distributions for given observations, and then compute the
        log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):
    
    """The network used by the value function."""
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape
        return torch.squeeze(self.v_net(obs), -1)



class MLPActorCritic(nn.Module):
    
    """Class to combine policy (actor) and value (critic) function neural networks."""

    def __init__(self,
                 hidden_sizes=(128,128), activation=nn.Tanh): #64,64
        super().__init__()

        obs_dim = 8

        # Build policy for 4-dimensional action space
        self.pi = MLPCategoricalActor(obs_dim, 4, hidden_sizes, activation)

        # Build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, state):
        """
        Take a state and return an action, value function, and log-likelihood
        of chosen action.
        """
        # TODO1: Implement this function.
        # It is supposed to return three numbers:
        #    1. An action sampled from the policy given a state (0, 1, 2 or 3)
        #    2. The value function at the given state
        #    3. The log-probability of the action under the policy output distribution
        # Hint: This function is only called when interacting with the environment. You should use
        # `torch.no_grad` to ensure that it does not interfere with the gradient computation.
        with torch.no_grad():
            pi, logp = self.pi.forward(torch.as_tensor(state, dtype=torch.float32))
            a = pi.sample() # torch.argmax(torch.Tensor([pi.log_prob(torch.Tensor([q])) for q in range(4)]))

            v = self.v.forward(torch.as_tensor(state, dtype=torch.float32))
            logp = pi.log_prob(a)

        
        return a, v, logp


class VPGBuffer:
   
    """
    Buffer to store trajectories.
    """
    def __init__(self, obs_dim, act_dim, size, gamma, lam):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # advantage estimates
        self.phi_buf = np.zeros(size, dtype=np.float32)
        # rewards
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # trajectory's remaining return
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # values predicted
        self.val_buf = np.zeros(size, dtype=np.float32)
        # log probabilities of chosen actions under behavior policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append a single timestep to the buffer. This is called at each environment
        update to store the outcome observed.
        """
        # buffer has to have room so you can store
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def end_traj(self, last_val=0):
        """
        Call after a trajectory ends. Last value is value(state) if cut-off at a
        certain state, or 0 if trajectory ended uninterrupted
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # TODO6: Implement computation of phi.
        
        # Hint: For estimating the advantage function to use as phi, equation 
        # 16 in the GAE paper (see task description) will be helpful, and so will
        # the discout_cumsum function at the top of this file. 
        #delete for a moment self.gamma*
        deltas = rews[:-1] + vals[1:] - vals[:-1]
        self.phi_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        #just this? 

        #TODO4: currently the return is the total discounted reward for the whole episode. 
        # Replace this by computing the reward-to-go for each timepoint.
        # Hint: use the discount_cumsum function.
        
        #if USE_NAIVE:
        #    self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[0] * np.ones(self.ptr-self.path_start_idx)
        #else:
        #    self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:self.ptr-self.path_start_idx]
        self.path_start_idx = self.ptr
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        #self.path_start_idx = self.ptr


    def get(self):
        """
        Call after an epoch ends. Resets pointers and returns the buffer contents.
        """
        # Buffer has to be full before you can get something from it.
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # TODO7: Here it may help to normalize the values in self.phi_buf
        self.phi_buf = (self.phi_buf - np.mean(self.phi_buf))/np.std(self.phi_buf)

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    phi=self.phi_buf, logp=self.logp_buf, val=self.val_buf) #val=self.val_buf added by us
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and not v[0].requiresgrad():
                print(v, 'does not require grad')
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class Agent:
    
    def __init__(self, env):
        self.env = env
        self.hid = 64 #64  # layer width of networks
        self.l = 2  # layer number of networks
        # initialises an actor critic
        self.ac = MLPActorCritic(hidden_sizes=[self.hid]*self.l)
        #added by us
        self.l0 = 4.7
        self.l1 = 5.79 
        self.l2 = 26.99 
        clip_ratio=0.2
        train_policy_iterations=80

        # Learning rates for policy and value function
        pi_lr = 3e-3#3e-3
        vf_lr = 1e-3#1e-3

        # we will use the Adam optimizer to update value function and policy
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        
    '''#let's try PPO!
    def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


    def logprobabilities(logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        num_actions= env.action_space.n#4 #???
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    def train_policy(self,observation_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            clip_ratio=0.2
            pi, logp = self.ac.pi.forward(torch.as_tensor(observation_buffer, dtype=torch.float32), torch.as_tensor(action_buffer, dtype=torch.float32))
            ratio = torch.exp(  #tf.exp
                logp ###self.logprobabilities(self.get_action(observation_buffer), action_buffer) actor(observation_buffer) as first argument
                - logprobability_buffer)
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,)
        
            self.pi_optimizer.zero_grad() #needed?
            policy_loss = -torch.Tensor(tf.reduce_mean(tf.minimum((ratio * advantage_buffer).detach.numpy(), min_advantage)))
            policy_loss = -torch.mean(tf.minimum((ratio * advantage_buffer), min_advantage))
            #policy_loss=torch.Tensor(policy_loss)
            policy_loss.backwards
            self.pi_optimizer.step()
        policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        kl = tf.reduce_mean(logprobability_buffer- logp ###logprobabilities(actor(observation_buffer), action_buffer))
        kl = tf.reduce_sum(kl)
        return torch.Tensor(kl) #kl'''

    def pi_update(self, data):
        """
        Use the data from the buffer to update the policy. Returns nothing.
        """
        #TODO2: Implement this function. 
        #TODO8: Change the update rule to make use of the baseline instead of rewards-to-go.
        #on policy, model free RL

        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']
        logp = data['logp'] #we added it
        '''train_policy_iterations=80
        #clip_ratio=0.2
        target_kl=0.01
        print(len(obs))
        print(len(act))
        print(len(logp))
        print(len(phi))'''
        '''# logp = self.ac.pi.get_likeklihood()
        #let's try PPO
        for _ in range(train_policy_iterations):
            kl = self.train_policy(obs, act, logp, phi)
            if kl > 1.5 * target_kl:
                # Early Stopping
                break'''

        #pi, logp = self.ac.pi.forward(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32)) # computed using the model to be optimized

        _, logp = self.ac.pi(obs, act)
        #Hint: you need to compute a 'loss' such that its derivative with respect to the policy
        #parameters is the policy gradient. Then call loss.backward() and pi_optimizer.step()

        self.pi_optimizer.zero_grad()  # reset the gradient in the policy optimizer
        
        good_action = logp*((act == 0).int()*self.l0 + self.l1*(act == 2).int())
        bad_action = self.l2*logp*((act == 1).int() + (act == 3).int())
        loss = -torch.mean(phi*logp - .001*(bad_action - good_action)) # last term encourages not moving
        #loss = -torch.mean((ret-phi)*logp - .001*(bad_action - good_action)) # is it bullshit now? YES!
        loss.backward()
        self.pi_optimizer.step()
        return

        # Before doing any computation, always call.zero_grad on the relevant optimizer
        '''self.pi_optimizer.zero_grad()

        #try it easier:
        _, logp = self.ac.pi(obs, act)
        phip = Variable(phi.squeeze())
        #phip=phi #choose which one to use
        loss = -(logp*phip).mean()
        loss.backward()
        self.pi_optimizer.step()
        return'''

    #wanted to make more elegant but not necessary
    #def compute_loss_pi(self, data):
    #    obs, act, phi = data['obs'], data['act'], data['phi']

    #    pi, logp = self.ac.pi(obs, act)
    #    loss_pi = -(logp * phi).mean()
    #    return loss_pi

    def v_update(self, data):
        """
        Use the data from the buffer to update the value function. Returns nothing.
        """
        #TODO5: Implement this function
        #TD LEARNING

        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']
        logp = data['logp']
        val = data['val'] #added by us

        # Hint: it often works well to do multiple rounds of value function updates per epoch.
        # With the learning rate given, we'd recommend 100. 
        # In each update, compute a loss for the value function, call loss.backwards() and 
        # then v_optimizer.step()
        # Before doing any computation, always call.zero_grad on the relevant optimizer

        '''old_states = obs[:-1]
        new_states = obs[1:]

        old_val = self.ac.v.forward(torch.as_tensor(new_states, dtype=torch.float32)).detach() # We use this multiple times so we detach it
        #our complex loss 
        loss_function = nn.MSELoss()
        for _ in range(100):
            self.v_optimizer.zero_grad()
            val = self.ac.v.forward(torch.as_tensor(old_states, dtype=torch.float32)) # new value

            loss = loss_function(.99*old_val + ret[:-1], val) / 2 # As with standard DQN but we cannot really control the actions as we would like...
            loss.backward(retain_graph=True)
            self.v_optimizer.step()

        return'''

        gamma=0.99
        loss_function = nn.MSELoss()
        #easy loss with only ret
        for _ in range(100):
            self.v_optimizer.zero_grad()

            # !!!!
            # target = torch.tensor(ret + gamma * phi) #bullshit I think
            v_value = self.ac.v.forward(obs)
            loss = loss_function(v_value, ret) #(not ret + gamma*old_value?)
            loss.backward()
            self.v_optimizer.step()

        return

        #attempt with old values from the buffer and new values by us (probably bullshit)
        '''for _ in range(100):
            self.v_optimizer.zero_grad()
            # compute a loss for the value function, call loss.backwards() and then v_optimizer.step()
            term2 = ret + 0.99*val
            #target = torch.tensor(target)
            term2.clone().detach().requires_grad_(True)
            v_val = self.ac.v(obs)
            v_loss = loss_function(v_val, term2)
            v_loss.backward()
            self.v_optimizer.step()
        return'''

    #analogous to compute loss pi, not necessary an auxiliary function 
    #def compute_loss_v(self, data, val):
    #    obs, ret = data['obs'], data['ret']
    #    return ((self.ac.v(obs) - ret)**2).mean()



    def train(self):
        """
        Main training loop.

        IMPORTANT: This function is called by the checker to train your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """

        # The observations are 8 dimensional vectors, and the actions are numbers,
        # i.e. 0-dimensional vectors (hence act_dim is an empty list).
        obs_dim = [8]
        act_dim = []

        # Training parameters
        # You may wish to change the following settings for the buffer and training
        # Number of training steps per epoch
        steps_per_epoch = 3000
        # Number of epochs to train for
        epochs = 50
        # The longest an episode can go on before cutting it off
        max_ep_len = 300
        # Discount factor for weighting future rewards
        gamma = 0.99
        lam = 0.97

        # Set up buffer
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the ADAM optimizer using the parameters
        # of the policy and then value networks

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0
        #try a "best update"
        current_best = -1e4
        
        # Main training loop: collect experience in env and update / log each epoch
        for epoch in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32)) # No grad

                next_state, r, terminal = self.env.transition(a.item())
                ep_ret += r
                ep_len += 1

                # Log transition
                buf.store(state, a, r, v, logp) # No grad

                # Update state (critical!)
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret)  # only store return when episode ended
                    buf.end_traj(v)
                    state, ep_ret, ep_len = self.env.reset(), 0, 0

            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            #to implement "best update"
             # If this return is the best until now, save the model so we can use it later
            if mean_return > current_best:
                current_best = mean_return
                joblib.dump(copy.deepcopy(self.ac), 'best_ac.pkl')
                print('updated best')

            
            if len(ep_returns) == 0:
                print(f"Epoch: {epoch+1}/{epochs}, all episodes exceeded max_ep_len")
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

            if PRUNE and mean_return > PRUNE_THRESHOLD:
                break # We're happy with this result

            # This is the end of an epoch, so here is where we update the policy and value function

            data = buf.get()

            self.pi_update(data)
            self.v_update(data)


        return True


    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function is called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        It is not used in your own training code. Instead the .step function in 
        MLPActorCritic is used since it also outputs relevant side-information. 
        """
        # TODO3: Implement this function.
        # Currently, this just returns a random action.
        pi, logp = self.ac.pi.forward(torch.as_tensor(obs, dtype=torch.float32))

        return pi.sample()

class OptimAgent:
    
    def __init__(self, env, l0, l1, l2):
        self.env = env
        self.hid = 64  # layer width of networks
        self.l = 2  # layer number of networks
        # initialises an actor critic
        self.ac = MLPActorCritic(hidden_sizes=[self.hid]*self.l)
        self.l0 = l0
        self.l1 = l1 
        self.l2 = l2 

        # Learning rates for policy and value function
        pi_lr = 3e-3#3e-3
        vf_lr = 1e-3#1e-3

        # we will use the Adam optimizer to update value function and policy
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    def pi_update(self, data):
        """
        Use the data from the buffer to update the policy. Returns nothing.
        """
        #TODO2: Implement this function. 
        #TODO8: Change the update rule to make use of the baseline instead of rewards-to-go.
        #on policy, model free RL

        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']
        # logp = data['logp'] #we added it
        # logp = self.ac.pi.get_likeklihood()

        pi, logp = self.ac.pi.forward(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32)) # computed using the model to be optimized

        #Hint: you need to compute a 'loss' such that its derivative with respect to the policy
        #parameters is the policy gradient. Then call loss.backward() and pi_optimizer.step()

        #pi_optimizer.zero_grad()  # reset the gradient in the policy optimizer
        
        good_action = logp*((act == 0).int()*self.l0 + self.l1*(act == 2).int())
        bad_action = self.l2*logp*((act == 1).int() + (act == 3).int())
        loss = -torch.mean(phi*logp - .001*(bad_action - good_action)) # last term encourages not moving, don't we use baselines anymore?
        loss.backward()
        self.pi_optimizer.step()
        return


    def compute_loss_pi(self, data):
        obs, act, phi = data['obs'], data['act'], data['phi']

        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * phi).mean()
        return loss_pi

    def v_update(self, data):
        """
        Use the data from the buffer to update the value function. Returns nothing.
        """
        #TODO5: Implement this function
        #TD LEARNING

        obs = data['obs']
        act = data['act']
        phi = data['phi']
        ret = data['ret']
        logp = data['logp']

        # Hint: it often works well to do multiple rounds of value function updates per epoch.
        # With the learning rate given, we'd recommend 100. 
        # In each update, compute a loss for the value function, call loss.backwards() and 
        # then v_optimizer.step()
        # Before doing any computation, always call.zero_grad on the relevant optimizer

        old_states = obs[:-1]
        new_states = obs[1:]

        old_val = self.ac.v.forward(torch.as_tensor(new_states, dtype=torch.float32)).detach() # We use this multiple times so we detach it

        loss_function = nn.MSELoss()
        for _ in range(100):
            self.v_optimizer.zero_grad()
            val = self.ac.v.forward(torch.as_tensor(old_states, dtype=torch.float32)) # new value

            loss = loss_function(.99*old_val + ret[:-1], val) / 2 # As with standard DQN but we cannot really control the actions as we would like...
            loss.backward(retain_graph=True)
            self.v_optimizer.step()

        return



    def compute_loss_v(self, data, val):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()



    def train(self):
        """
        Main training loop.

        IMPORTANT: This function is called by the checker to train your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        """

        # The observations are 8 dimensional vectors, and the actions are numbers,
        # i.e. 0-dimensional vectors (hence act_dim is an empty list).
        obs_dim = [8]
        act_dim = []

        # Training parameters
        # You may wish to change the following settings for the buffer and training
        # Number of training steps per epoch
        steps_per_epoch = 3000
        # Number of epochs to train for
        epochs = 50
        # The longest an episode can go on before cutting it off
        max_ep_len = 300
        # Discount factor for weighting future rewards
        gamma = 0.99
        lam = 0.97

        # Set up buffer
        buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

        # Initialize the ADAM optimizer using the parameters
        # of the policy and then value networks

        # Initialize the environment
        state, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main training loop: collect experience in env and update / log each epoch
        for epoch in range(epochs):
            ep_returns = []
            for t in range(steps_per_epoch):
                a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32)) # No grad

                next_state, r, terminal = self.env.transition(a.item())
                ep_ret += r
                ep_len += 1

                # Log transition
                buf.store(state, a, r, v, logp) # No grad

                # Update state (critical!)
                state = next_state

                timeout = ep_len == max_ep_len
                epoch_ended = (t == steps_per_epoch - 1)

                if terminal or timeout or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
                    else:
                        v = 0
                    if timeout or terminal:
                        ep_returns.append(ep_ret)  # only store return when episode ended
                    buf.end_traj(v)
                    state, ep_ret, ep_len = self.env.reset(), 0, 0

            mean_return = np.mean(ep_returns) if len(ep_returns) > 0 else np.nan
            if len(ep_returns) == 0:
                print(f"Epoch: {epoch+1}/{epochs}, all episodes exceeded max_ep_len")
            print(f"Epoch: {epoch+1}/{epochs}, mean return {mean_return}")

            if PRUNE and mean_return > PRUNE_THRESHOLD:
                break # We're happy with this result

            # This is the end of an epoch, so here is where we update the policy and value function

            data = buf.get()

            self.pi_update(data)
            self.v_update(data)


        return True


    def get_action(self, obs):
        """
        Sample an action from your policy.

        IMPORTANT: This function is called by the checker to evaluate your agent.
        You SHOULD NOT change the arguments this function takes and what it outputs!
        It is not used in your own training code. Instead the .step function in 
        MLPActorCritic is used since it also outputs relevant side-information. 
        """
        # TODO3: Implement this function.
        # Currently, this just returns a random action.
        #pi, logp = self.ac.pi.forward(torch.as_tensor(obs, dtype=torch.float32))

        return self.ac.pi(torch.tensor(np.array([obs])).float())[0].sample()[0].item()
        #return #pi.sample()


'''def objective(l0, l1, l2):
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    trial_results = []

    for trial_nr in range(10):
        env = LunarLander()
        env.seed(seed=89279)

        agent = OptimAgent(env, l0, l1, l2)
        agent.train()

        episode_length = 300
        n_eval = 100
        returns = []
         print("Evaluating agent...")

        for i in range(n_eval):
           print(f"Testing policy: episode {i+1}/{n_eval}")
            state = env.reset()
            cumulative_return = 0
            # The environment will set terminal to True if an episode is done.
            terminal = False
            env.reset()
            for t in range(episode_length):
                # Taking an action in the environment
                action = agent.get_action(state)
                state, reward, terminal = env.transition(action.item())
                cumulative_return += reward
                if terminal:
                    break
            returns.append(cumulative_return)
            # print(f"Achieved {cumulative_return:.2f} return.")
        env.close()
        trial_results = trial_results + [np.mean(returns)]

    return -np.mean(trial_results)'''


def main():
    """
    Train and evaluate agent.

    This function basically does the same as the checker that evaluates your agent.
    You can use it for debugging your agent and visualizing what it does.
    """
    from lunar_lander import LunarLander
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = LunarLander()
    env.seed(0)

    agent = Agent(env)
    agent.train()

    rec = VideoRecorder(env, "policy.mp4")
    episode_length = 300
    n_eval = 100
    returns = []
    print("Evaluating agent...")

    for i in range(n_eval):
        print(f"Testing policy: episode {i+1}/{n_eval}")
        state = env.reset()
        cumulative_return = 0
        # The environment will set terminal to True if an episode is done.
        terminal = False
        env.reset()
        for t in range(episode_length):
            if i <= 10:
                rec.capture_frame()
            # Taking an action in the environment
            action = agent.get_action(state)
            state, reward, terminal = env.transition(action.item())
            cumulative_return += reward
            if terminal:
                break
        returns.append(cumulative_return)
        print(f"Achieved {cumulative_return:.2f} return.")
        if i == 10:
            rec.close()
            print("Saved video of 10 episodes to 'policy.mp4'.")
    env.close()
    print(f"Average return: {np.mean(returns):.2f}")

if __name__ == "__main__":
    main()
