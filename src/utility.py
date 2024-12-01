from src.MAB import MAB
import numpy as np
import matplotlib.pyplot as plt

def random_argmax(a):
    """
    Select the index corresponding to the maximum in the input list.
    Ties are randomly broken.
    """
    return np.random.choice(np.where(a == a.max())[0])
    
def run_algorithm(mab, alg, R=10, reset = True):
    T = mab.get_T()
    regret_list = np.zeros((R, T))  # To store regrets for each repetition
    for r in range(R):
        for t in range(T):
            alg.play_one_step()  # Run one step of the algorithm
        regret_list[r, :] = mab.get_regrets()  # Collect regrets
        alg.reset()  # Reset algorithm for the next repetition
        if reset:
            mab.reset()
    return regret_list

def plot_regret_single(mab, regrets, alg_name):
    T = mab.get_T()
    mean_regret = np.mean(regrets, axis=0)
    lower_bound = np.percentile(regrets, 5, axis=0)
    upper_bound = np.percentile(regrets, 95, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(range(T), mean_regret, label="Mean Regret")
    plt.fill_between(
        range(T), lower_bound, upper_bound, color="b", alpha=0.1, label="90% CI"
    )
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Cumulative Regret: {alg_name}")
    plt.axvline(mab._T_resampled, linestyle = '--', color = 'red')
    plt.axhline(mean_regret[-1], linestyle = '--', color = 'grey', alpha = 0.5)
    plt.text(
        T - 1, mean_regret[-1] + 0.1,  # Position of the label (slightly above the line)
        f"{mean_regret[-1]:.2f}",       # Format the value to 2 decimal places
        color='grey', fontsize=10, ha='center', va='bottom'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('mean regret: {:.2f}'.format(mean_regret[-1]))
    print('5% regret: {:.2f}'.format(lower_bound[-1]))
    print('95% regret: {:.2f}'.format(upper_bound[-1]))
    
def plot_regret_double(mab, regrets_1, regrets_2, alg_name_1, alg_name_2):
    T = mab.get_T()
    
    # Compute mean and confidence intervals for both algorithms
    mean_regret_1 = np.mean(regrets_1, axis=0)
    lower_bound_1 = np.percentile(regrets_1, 5, axis=0)
    upper_bound_1 = np.percentile(regrets_1, 95, axis=0)
    
    mean_regret_2 = np.mean(regrets_2, axis=0)
    lower_bound_2 = np.percentile(regrets_2, 5, axis=0)
    upper_bound_2 = np.percentile(regrets_2, 95, axis=0)

    plt.figure(figsize=(8, 6))
    
    # Plot mean regrets for both algorithms
    plt.plot(range(T), mean_regret_1, label=f"Mean Regret: {alg_name_1}")
    plt.plot(range(T), mean_regret_2, label=f"Mean Regret: {alg_name_2}")

    # Fill the areas for 90% CI for both algorithms
    plt.fill_between(range(T), lower_bound_1, upper_bound_1, color="b", alpha=0.1)
    plt.fill_between(range(T), lower_bound_2, upper_bound_2, color="orange", alpha=0.1)

    # Add labels and title
    plt.xlabel("Step")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Comparison of Cumulative Regret: {alg_name_1} vs {alg_name_2}")
    
    # Add vertical line for re-sampling point
    plt.axvline(mab._T_resampled, linestyle='--', color='red')
    
    # Add horizontal lines at the final mean regret values for both algorithms
    plt.axhline(mean_regret_1[-1], linestyle='--', color='blue', alpha=0.5)
    plt.axhline(mean_regret_2[-1], linestyle='--', color='orange', alpha=0.5)
    
    # Add numerical labels above the horizontal lines
    plt.text(
        T - 1, mean_regret_1[-1] + 0.1,  # Position of the label for algorithm 1
        f"{mean_regret_1[-1]:.2f}",       # Format the value to 2 decimal places
        color='blue', fontsize=10, ha='center', va='bottom'
    )
    plt.text(
        T - 1, mean_regret_2[-1] + 0.1,  # Position of the label for algorithm 2
        f"{mean_regret_2[-1]:.2f}",       # Format the value to 2 decimal places
        color='orange', fontsize=10, ha='center', va='bottom'
    )

    # Add legend
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_regret_grid(mab, regrets_list, alg_names):
    """
    Plots a 2x3 grid of cumulative regret plots.
    
    Parameters:
    - mab: The multi-armed bandit object containing the time horizon (T) and resampling step.
    - regrets_list: List of 6 numpy arrays of regrets.
    - alg_names: List of 6 algorithm names corresponding to the regrets.
    """
    T = mab.get_T()
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    
    for i, ax in enumerate(axs.flat):
        if i >= len(regrets_list):
            break  # Safety check if regrets_list is less than 6

        regrets = regrets_list[i]
        alg_name = alg_names[i]
        
        mean_regret = np.mean(regrets, axis=0)
        lower_bound = np.percentile(regrets, 5, axis=0)
        upper_bound = np.percentile(regrets, 95, axis=0)
        
        ax.plot(range(T), mean_regret, label="Mean Regret")
        ax.fill_between(
            range(T), lower_bound, upper_bound, color="b", alpha=0.1, label="90% CI"
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f"Cumulative Regret: {alg_name}")
        ax.axvline(mab._T_resampled, linestyle='--', color='red')
        ax.axhline(mean_regret[-1], linestyle='--', color='grey', alpha=0.5)
        ax.text(
            T - 1, mean_regret[-1] + 0.1,  # Position of the label
            f"{mean_regret[-1]:.2f}",     # Format the value
            color='grey', fontsize=10, ha='center', va='bottom'
        )
        ax.legend()
    
    plt.tight_layout()
    plt.show()

