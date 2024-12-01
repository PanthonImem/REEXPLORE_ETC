import numpy as np

class MAB:
    def __init__(self, T=2000, K=5, T_resampled=500, mu_low=0, mu_high=10, std=1, portion=1):
        '''
        Parameters:
            T: horizon (number of steps)
            K: number of bandits (default 10)
            T_resampled: Number of steps before perturbing the bandits' means
            mu_low: Minimum value for bandit means
            mu_high: Maximum value for bandit means
            std: Standard deviation for reward noise
            portion: Portion of arms to perturb when resampling
        '''
        self.__K = K
        self.__mu_list = np.random.uniform(mu_low, mu_high, size=K)
        self._std = std
        self.__T = T
        self._T_resampled = T_resampled
        self.__reward = np.zeros(K)  # Total reward for each bandit
        self.__reward_list = []  # List to track the reward history at each step
        self.__regrets = []  # List to track the regret history at each step
        self.mu_low = mu_low
        self.mu_high = mu_high
        self.portion = portion
        self.num_pulls = 0
        self.num_pulls_by_arm = np.zeros(K)

        # New: Store rewards for each arm (list of lists)
        self.__rewards_by_arm = [[] for _ in range(K)]

        #prepare set 2 of mu for after change in reward
        self.__mu_list2 = self.__mu_list.copy()
        perturbed_list = np.random.choice(self.__K, int(self.portion * self.__K), replace=False)
        for ind in perturbed_list:
            self.__mu_list2[ind] = np.random.uniform(self.mu_low, self.mu_high)
            
    def pull(self, ind):
        '''
        Pull the bandit with index ind and store the reward
        '''
        reward = np.random.normal(self.__mu_list[ind], self._std)  # Normal distribution with mean mu and std 1
        self.__reward[ind] += reward
        self.__regrets.append(max(self.__mu_list) - self.__mu_list[ind])
        self.num_pulls += 1
        self.num_pulls_by_arm[ind] += 1

        # Store the reward for the pulled arm
        self.__rewards_by_arm[ind].append(reward)

        # Half-way perturbation of bandit means
        if self.num_pulls == self._T_resampled:
            self.__mu_list = self.__mu_list2

        # Append the current total reward to the reward history
        self.__reward_list.append(reward)

        return reward

    def reset(self):
        '''
        Reset the bandit to its initial state.
        '''
        # Reset total rewards for each bandit
        self.__reward = np.zeros(self.__K)
        self.__reward_list = []  # Clear the reward history
        self.__regrets = []  # Clear the regret history
        self.num_pulls = 0  # Reset total number of pulls
        self.num_pulls_by_arm = np.zeros(self.__K)  # Reset pulls for each arm
        
        # Resample the initial means for the bandits
        self.__mu_list = np.random.uniform(self.mu_low, self.mu_high, size=self.__K)
        
        # Resample the second set of means for after perturbation
        self.__mu_list2 = self.__mu_list.copy()
        perturbed_list = np.random.choice(self.__K, int(self.portion * self.__K), replace=False)
        for ind in perturbed_list:
            self.__mu_list2[ind] = np.random.uniform(self.mu_low, self.mu_high)
        
        # Reset rewards history for each arm
        self.__rewards_by_arm = [[] for _ in range(self.__K)]

    def get_reward(self):
        '''
        Get the current total rewards for each bandit
        '''
        return self.__reward

    def get_reward_list(self):
        '''
        Get the reward history for all steps
        '''
        return self.__reward_list

    def get_record_list(self):
        '''
        Get the history of rewards for all steps (list of records)
        '''
        return self.__reward_list

    def get_regrets(self):
        '''
        Get cumulative regrets for all steps
        '''
        return np.cumsum(self.__regrets)

    def get_T(self):
        '''
        Get the time horizon (number of steps)
        '''
        return self.__T

    def get_K(self):
        '''
        Get the number of bandits (arms)
        '''
        return self.__K

    def get_mu_list(self):
        '''
        Get the true mean values of the bandits
        '''
        return self.__mu_list

    def get_mu_list2(self):
        '''
        Get the true mean values of the bandits
        '''
        return self.__mu_list2

    def get_rewards_by_arm(self):
        '''
        Get the list of rewards for each arm
        '''
        return self.__rewards_by_arm
