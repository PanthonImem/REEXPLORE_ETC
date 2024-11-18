import numpy as np
from scipy.stats import ttest_ind
from src.MAB import MAB
from src.utility import random_argmax

class REEXPLORE:
    def __init__(self, MAB, T_explore, T_reexplore, epsilon=0.001, M=50, cooldown=500):
        """
        RE-EXPLORE Algorithm implementation with cooldown period.
        
        :param MAB: The multi-armed bandit instance to interact with.
        :param T_explore: The number of steps for the exploration phase.
        :param T_reexplore: The number of steps for the re-exploration phase.
        :param epsilon: Threshold for the t-test.
        :param M: The size of the reward history for comparison.
        :param cooldown: Number of steps to wait after a re-exploration phase before allowing another.
        """
        self.MAB = MAB
        self.T_explore = T_explore
        self.T_reexplore = T_reexplore
        self.epsilon = epsilon
        self.M = M
        self.cooldown = cooldown
        self.reexplore_count = 0
        self.cooldown_count = 0
        self.arm_reward_explore = [[] for _ in range(self.MAB.get_K())]
        self.num_pulls_by_arm_explore = np.zeros(self.MAB.get_K())

    def reset(self):
        """Resets the environment and state."""
        self.MAB.reset()
        self.reexplore_count = 0
        self.cooldown_count = 0
        self.arm_reward_explore = [[] for _ in range(self.MAB.get_K())]
        self.num_pulls_by_arm_explore = np.zeros(self.MAB.get_K())

    def play_one_step(self):
        K = self.MAB.get_K()
        i = self.MAB.num_pulls
        Ne = self.T_explore // K  # Number of pulls per arm in the exploration phase
        pulls = self.MAB.num_pulls_by_arm

        # Handle cooldown period after reexploration
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return self.commit_to_best_arm()

        # Step 1: Exploration Phase
        if np.min(pulls) < Ne or (0 < self.reexplore_count < self.T_reexplore):
            if self.reexplore_count > 0:
                self.reexplore_count += 1

            underexplored_arms = np.where(pulls < Ne)[0]
            a = np.random.choice(underexplored_arms if underexplored_arms.size > 0 else np.arange(K))
            reward = self.MAB.pull(a)
            self.arm_reward_explore[a].append(reward)
            self.num_pulls_by_arm_explore[a] += 1
            return reward

        # Step 2: Commit to the best arm based on average reward
        if self.reexplore_count == self.T_reexplore:
            self.reexplore_count = 0
            self.cooldown_count = self.cooldown  # Start cooldown period
            return self.commit_to_best_arm()

        # Step 3: Check if re-exploration is needed
        arm_profile = self.arm_reward_explore[self.get_best_arm()]
        latest_rewards_a_star = self.MAB.get_rewards_by_arm()[self.get_best_arm()][-self.M:]

        if len(latest_rewards_a_star) >= Ne:
            _, p_value = ttest_ind(latest_rewards_a_star, arm_profile, equal_var=True)
            if p_value < self.epsilon:
                #print('reexplore', i)
                self.trigger_reexploration()
                return self.explore_random_arm()

        # Step 4: Commit to the best arm and return reward
        return self.commit_to_best_arm()

    def get_best_arm(self):
        """Select the arm with the highest average reward."""
        total_rewards = np.array([np.sum(rewards) for rewards in self.arm_reward_explore])
        avg_rewards = np.divide(total_rewards, self.num_pulls_by_arm_explore, where=self.num_pulls_by_arm_explore != 0)
        return random_argmax(avg_rewards)

    def commit_to_best_arm(self):
        """Commit to the best arm and return the reward."""
        a_star = self.get_best_arm()
        reward = self.MAB.pull(a_star)
        self.arm_reward_explore[a_star].append(reward)
        self.num_pulls_by_arm_explore[a_star] += 1
        return reward

    def explore_random_arm(self):
        """Explore a random arm."""
        K = self.MAB.get_K()
        a = np.random.choice(K)
        reward = self.MAB.pull(a)
        self.arm_reward_explore[a].append(reward)
        self.num_pulls_by_arm_explore[a] += 1
        return reward

    def trigger_reexploration(self):
        """Trigger re-exploration by resetting relevant variables."""
        self.reexplore_count = 1
        self.arm_reward_explore = [[] for _ in range(self.MAB.get_K())]