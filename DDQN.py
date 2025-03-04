import numpy as np
import random
from SumTree import SumTree
from keras import backend as K
from matplotlib import pyplot as plt

                  
class DoubleDQNAgent:

    def __init__(self, online_network, target_network, **kwargs):
        # Initialize parameters
        self.online_network = online_network
        self.target_network = target_network
        self.rowdim = kwargs['ROWDIM']
        self.coldim = kwargs['COLDIM']
        self.gamma = kwargs['GAMMA']
        self.epsilon = kwargs['EPSILON_INITIAL']
        self.epsilon_decay = kwargs['EPSILON_DECAY']
        self.epsilon_min = kwargs['EPSILON_MIN']
        self.tau = kwargs['TAU']
        self.batch_size = kwargs['EXPERIENCE_REPLAY_BATCH_SIZE']                        
        self.memory_limit = kwargs['AGENT_MEMORY_LIMIT']
        self.num_holdout_states = kwargs['NUM_HOLDOUT_STATES']
        self.per_alpha = kwargs['PER_ALPHA']
        self.per_beta_min = kwargs['PER_BETA_MIN']
        self.per_beta_max = kwargs['PER_BETA_MAX']
        self.per_beta_anneal_steps = kwargs['PER_BETA_ANNEAL_STEPS']
        self.per_epsilon = kwargs['PER_EPSILON']
        self.lr_piecewise = kwargs['LR_PIECEWISE']
        self.lr_decay_steps = kwargs['LR_DECAY_STEPS']

        self.steps = 0
        self.holdout_states = []

        self.lrate = self.lr_piecewise[0]
        self.lrate_decay = []
        for idx in range(0,len(self.lr_piecewise)-1):
            self.lrate_decay.append((self.lr_piecewise[idx] - self.lr_piecewise[idx+1]) \
                                    / (self.lr_decay_steps[idx+1]-self.lr_decay_steps[idx]))
        # Prioritized Experience Replay (PER) parameters
        self.beta_anneal = (self.per_beta_max - self.per_beta_min) / self.per_beta_anneal_steps
        self.per_beta = self.per_beta_min
        self.sumtree = SumTree(self.memory_limit)
        self.memory_length = 0
        
    
    def act(self, state):
        flattened_state = state.flatten()
        nn_state = self.reshape_state_for_net(state)
        if self.epsilon > np.random.rand():
            # Explore but only choose hidden tiles (#9)
            valid_actions = np.where(flattened_state == 9)[0]
            return np.random.choice(valid_actions), nn_state, valid_actions
        else:
            # Exploit, but only choose hidden tiles (#9)
            valid_actions = [0 if x == 9 else 1 for x in flattened_state]

            q_values = self.online_network.predict(nn_state)

            valid_qvalues = np.ma.masked_array(q_values, valid_actions)
            return np.argmax(valid_qvalues), nn_state, np.squeeze(valid_qvalues)


    def experience_replay(self):
        select_network = self.online_network
        eval_network = self.target_network
        
        minibatch, tree_indices, weights = self._per_sample()
        minibatch_new_q_values = []

        for experience, tree_idx in zip(minibatch, tree_indices):
            state, action, reward, next_state, done, nn_state, nn_next_state = experience
            experience_new_q_values = select_network.predict(nn_state)[0]
            if done:
                q_update = reward
            else:
                valid_actions = [0 if x == 9 else 1 for x in next_state.flatten()]

                # SELECT function
                predicted_qvalues = select_network.predict(nn_next_state)[0]
                select_net_selected_action = np.argmax(np.ma.masked_array(predicted_qvalues, valid_actions))
                
                # EVAL function
                eval_net_evaluated_q_value = eval_network.predict(nn_next_state)[0][select_net_selected_action]
                q_update = reward + self.gamma * eval_net_evaluated_q_value
            
            # Update sum tree 
            td_error = experience_new_q_values[action] - q_update
            td_error = np.clip(td_error, -1, 1) # Clip for stability
            priority = (np.abs(td_error) + self.per_epsilon)  ** self.per_alpha
            self.sumtree.update(tree_idx, priority)
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.squeeze(np.array([e[5] for e in minibatch]))
        minibatch_new_q_values = np.array(minibatch_new_q_values, dtype=np.float64)
        
        select_network.train_on_batch(minibatch_states, minibatch_new_q_values, sample_weight=weights)

        K.set_value(select_network.optimizer.learning_rate, self.lrate_decay_callback())


    def _per_sample(self):
        minibatch = []
        tree_indices = []
        priorities = []
        weights = []

        samples_per_segment = self.sumtree.total() / self.batch_size
        for segment in range(0,self.batch_size):
            seg_start = samples_per_segment * segment
            seg_end = samples_per_segment * (segment + 1)
            sample = random.uniform(seg_start, seg_end)
            (tree_index, priority, experience) = self.sumtree.get(sample)
            tree_indices.append(tree_index)
            priorities.append(priority)
            minibatch.append(experience)
        
        # Calculate and scale weights for importance sampling
        min_probability = np.min(priorities) / self.sumtree.total()
        max_weight = (min_probability * self.memory_length) ** (-self.per_beta)
        for priority in priorities:
            probability = priority / self.sumtree.total()
            weight = (probability * self.memory_length) ** (-self.per_beta)
            weights.append(weight / max_weight)
            
        return minibatch, tree_indices, np.array(weights)
    

    def remember(self, state, action, reward, next_state, done, nn_state):
        nn_next_state = self.reshape_state_for_net(next_state)
        priority = 1
        experience = (state, action, reward, next_state, done, nn_state, nn_next_state)
        self.sumtree.add(priority, experience)

        if self.memory_length < self.memory_limit: self.memory_length += 1
        # Make copies of the initial states as a holdout set to monitor convergence
        if len(self.holdout_states) < self.num_holdout_states:
            self.holdout_states.append(nn_state)

    
    def reshape_state_for_net(self, state):
        batch_size = 1
        nn_input = np.zeros((batch_size,self.rowdim, self.coldim, 9))
        
        for tile_num in range(0,9):
            idx1, idx2 = np.where(state == tile_num)
            nn_input[0, idx1, idx2, tile_num] = 1
        
        return nn_input

    def lrate_decay_callback(self):
        # Decays NN learning rate in a piecewise-linear fashion during training
        lr_ds = self.lr_decay_steps

        cond_list = []
        for idx in range(0, len(lr_ds)-1):
            cond_list.append(self.steps >= lr_ds[idx] and self.steps < lr_ds[idx+1])

        func_list = [lambda x=self.steps, lr=a, step_offset=b, decay=c: \
                  lr - (x-step_offset) * decay \
                      for a, b, c in zip(self.lr_piecewise[0:-1], lr_ds[0:-1], self.lrate_decay)]
        func_list.append(self.lr_piecewise[-1]) 
        
        self.lrate = float(np.piecewise(float(self.steps), cond_list, func_list))
        return self.lrate


    def update_beta(self):
        self.per_beta = min(self.per_beta + self.beta_anneal, self.per_beta_max)
        
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


    def update_target_network(self):
        online_network_weights = self.online_network.get_weights()
        target_network_weights = self.target_network.get_weights()
        layer_idx = 0
        for online_weight, target_weight in zip(online_network_weights,target_network_weights):
            updated_weight = target_weight * (1-self.tau) + online_weight * self.tau
            target_network_weights[layer_idx] = updated_weight
            layer_idx += 1
        self.target_network.set_weights(target_network_weights)
        
    
    def test_lrate_decay(self):
        current_step = self.steps # Keep track of agent's current step count
        current_lrate = self.lrate # Keep track of agent's current learn rate
        lr_ds = self.lr_decay_steps
        numpts = [2, 11, 11]
        for plot_type in range(0,3):
            step_list = []
            for idx in range(0,len(lr_ds)-1):
                if plot_type == 2:
                    step_list.append(np.linspace(lr_ds[idx], lr_ds[idx+1], num=numpts[plot_type]))
                    if idx == len(lr_ds)-2:
                        step_list.append(np.linspace(lr_ds[-1], lr_ds[-1]*1.5, num=numpts[plot_type]))
                else:
                    step_list.extend(np.linspace(lr_ds[idx], lr_ds[idx+1], num=numpts[plot_type]))
                    if idx == len(lr_ds)-2:
                        step_list.extend(np.linspace(lr_ds[-1], lr_ds[-1]*1.5, num=numpts[plot_type]))
            lrate = []
            if plot_type == 2:
                for sub_list in step_list:
                    temp_list = []
                    for step in sub_list:
                       self.steps = step 
                       temp_list.append(self.lrate_decay_callback()) 
                    lrate.append(temp_list)
            else:
                for step in step_list: # Sweep through steps
                    self.steps = step 
                    lrate.append(self.lrate_decay_callback())
            
        return step_list, lrate