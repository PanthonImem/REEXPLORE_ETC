import numpy as np
from src.MAB import MAB
from src.utility import random_argmax

def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])


class Explore:
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        pulls = self.MAB.num_pulls_by_arm
        least_pulled_arms = np.where(pulls == pulls.min())[0]
        a = np.random.choice(least_pulled_arms)
        reward = self.MAB.pull(a)
        return reward

class Greedy:
    def __init__(self, MAB):
        self.MAB = MAB

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        pulls = self.MAB.num_pulls_by_arm  # Number of pulls for each arm
        rewards = self.MAB.get_reward()  # Total rewards for each arm

        # Exploration: Play an unpulled arm if any exist
        if np.any(pulls == 0):
            unpulled_arms = np.where(pulls == 0)[0]
            chosen_arm = np.random.choice(unpulled_arms)
        else:
            # Exploitation: Play the arm with the highest average reward
            avg_rewards = np.divide(rewards, pulls, where=pulls != 0)  # Avoid divide by zero
            chosen_arm = random_argmax(avg_rewards)

        # Pull the chosen arm and return the reward
        return self.MAB.pull(chosen_arm)

class ETC:
    def __init__(self, MAB, Ne = 30):
        self.MAB = MAB
        self.Ne = Ne
        self.committed_arm = None

    def reset(self):
        self.MAB.reset()
        self.committed_arm = None

    def play_one_step(self):
        Ne = self.Ne
        pulls = self.MAB.num_pulls_by_arm

        # Exploration Phase
        if np.min(pulls) < Ne:
            arms_to_explore = np.where(pulls < Ne)[0]
            a = np.random.choice(arms_to_explore)
            reward = self.MAB.pull(a)
            return reward

        # Commit Phase
        if self.committed_arm is None:
            rewards = self.MAB.get_reward()
            avg_rewards = np.divide(rewards, pulls, where=pulls != 0)
            self.committed_arm = random_argmax(avg_rewards)

        return self.MAB.pull(self.committed_arm)

class Epgreedy:
    def __init__(self, MAB, delta=0.1):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        rewards = self.MAB.get_reward()
        pulls = self.MAB.num_pulls_by_arm
        
        T = self.MAB.get_T()
        K = self.MAB.get_K()
        
        if np.min(pulls) == 0:
            a = np.where(pulls == 0)[0]
            reward = self.MAB.pull(np.random.choice(a))
            return reward
        else:
            epsilon_t = np.minimum((K*np.log(T)/T)**(1/3),1)
            if(np.random.uniform(0,1)<epsilon_t):
                a = np.arange(0, len(pulls))
                reward = self.MAB.pull(np.random.choice(a))
                return reward
            else:
                avg_rewards = np.divide(rewards, pulls, where=pulls != 0)
                a = np.where(avg_rewards == avg_rewards.max())[0]
                return self.MAB.pull(np.random.choice(a)) 

class UCB:
    def __init__(self, MAB, delta=0.05):
        self.MAB = MAB
        self.delta = delta

    def reset(self):
        self.MAB.reset()

    def play_one_step(self):
        rewards = self.MAB.get_reward()
        pulls = self.MAB.num_pulls_by_arm
        T = self.MAB.get_T()
        K = self.MAB.get_K()

        ucb_array = np.zeros(K)
        avg_rewards = np.divide(rewards, pulls, where=pulls != 0)
        for i in range(K):
            ucb_array[i] = avg_rewards[i] + np.sqrt(np.log(K*T/self.delta)/(pulls[i]+1e-3))
        a = np.where(ucb_array == ucb_array.max())[0]
        return self.MAB.pull(np.random.choice(a)) 

class Thompson_sampling:
    def __init__(self, MAB, prior_mu=5, prior_sigma_sq=1):
        self.MAB = MAB
        self.K = self.MAB.get_K()  # Number of arms
        self.prior_mu = prior_mu  # Prior mean
        self.prior_sigma_sq = prior_sigma_sq  # Prior variance
        
        # Initialize mean and variance estimates for each arm
        self.mu = np.ones(self.K) * prior_mu
        self.sigma_sq = np.ones(self.K) * prior_sigma_sq
        
        # Initialize the number of pulls and rewards for each arm
        self.n = np.zeros(self.K)  # Number of pulls for each arm
        self.sums = np.zeros(self.K)  # Sum of rewards for each arm

    def reset(self):
        """Resets the Thompson Sampling algorithm."""
        self.MAB.reset()
        self.mu = np.ones(self.K) * self.prior_mu
        self.sigma_sq = np.ones(self.K) * self.prior_sigma_sq
        self.n = np.zeros(self.K)
        self.sums = np.zeros(self.K)

    def play_one_step(self):
        """Plays one step of Thompson Sampling."""
        # Sample from the posterior for each arm
        theta_samples = np.array([
            np.random.normal(self.mu[k], np.sqrt(self.sigma_sq[k])) for k in range(self.K)
        ])

        # Choose the arm with the highest sample
        a = np.argmax(theta_samples)

        # Pull the chosen arm and get the reward
        reward = self.MAB.pull(a)

        # Update the posterior parameters (mean and variance)
        self.n[a] += 1
        self.sums[a] += reward
        
        # Update mean (posterior mean) and variance (posterior variance)
        new_mu = (self.sums[a] + self.prior_mu / self.prior_sigma_sq) / (self.n[a] + 1 / self.prior_sigma_sq)
        new_sigma_sq = 1 / (self.n[a] + 1 / self.prior_sigma_sq)

        # Update the parameters for the arm
        self.mu[a] = new_mu
        self.sigma_sq[a] = new_sigma_sq

        return reward
