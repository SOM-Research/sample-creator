from collections import Counter
from scipy.stats import norm
import math

def extract_population_size_and_means(statistics):
    """
    Extract the population size and means from the statistics dictionary.
    Handles both numerical and categorical variables.

    Args:
    - statistics (dict): A dictionary containing statistical information for each variable.

    Returns:
    - tuple: A tuple containing the population size (N) and a list of means (mu) for each numerical variable. For categorical variables, only the population size is included.
    """
    N = None
    mu = []

    for variable_name, stats in statistics.items():
        if N is None:
            N = stats['Population Size']  # Assuming the population size is the same for all variables

        # Check if the variable is numerical and has a 'Mean' key
        if 'Mean' in stats:
            mu.append(stats['Mean'])

    return N, mu

def nis_phi(classified_observations, N): 
    """
    Calculate the number of observations in each stratum and their proportions with respect to the total population.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum (as a string) and each value is a list of observations.
    - N (int): The total number of observations.

    Returns:
    - tuple: A tuple containing two lists:
             1. A list containing the number of observations in each stratum (nis).
             2. A list containing the proportion of each stratum with respect to the total population (phi).
    """
    # Calculate the number of observations in each stratum
    nis = [len(obs_list) for obs_list in classified_observations.values()]

    # Calculate the proportion of each stratum with respect to the total population
    phi = [ni / N for ni in nis]

    return nis, phi

def sample_size(epsilon, confidence): # ONLY FOR CAT
    """
    Calculates the required sample size (n) given the precision (epsilon) and confidence level.

    Parameters:
    epsilon (float): Desired precision.
    confidence (float): Confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    n (int): Required sample size (rounded up).
    """
    alfa = 1 - confidence
    za = norm.ppf(1 - alfa / 2)
    n = (za / (2 * epsilon)) ** 2
    return math.ceil(n)

def determine_ni_size(phi, combination_strata, n): 
    """
    Calculate the sample size for each stratum based on proportions and the desired total sample size.

    Args:
    - phi (list): A list containing the proportion of each stratum with respect to the total population.
    - combination_strata (list): A list containing the combinations (strata) as keys.
    - n (int): The desired total sample size.

    Returns:
    - dict: A dictionary where each key is a stratum (as a string) and each value is the calculated sample size for that stratum.
    """
    K = len(phi)  # Number of strata

    # Initialize the dictionary to store the sample size for each stratum
    n_stratum = {}
    total_allocated = 0

    # Calculate the initial sample size for each stratum
    for i, proportion in enumerate(phi):
        stratum_key = f"({combination_strata[i][0]}, {combination_strata[i][1]})"
        ni = round(proportion * n)  # Calculate the sample size for the current stratum
        n_stratum[stratum_key] = ni  # Store the sample size in the dictionary
        total_allocated += ni

    # Calculate the difference between the total allocated and the desired total sample size
    difference = n - total_allocated

    # Distribute the difference proportionally among the strata
    if difference != 0:
        for i, proportion in enumerate(phi):
            stratum_key = f"({combination_strata[i][0]}, {combination_strata[i][1]})"
            additional_allocation = round(proportion * difference)
            n_stratum[stratum_key] += additional_allocation

    return n_stratum

def create_sample(n_stratum, classified_observations): 
    """
    Create a sample based on the provided sample sizes for each stratum and the classified observations.

    Args:
    - n_stratum (dict): A dictionary where each key is a stratum (as a string) and each value is the calculated sample size for that stratum.
    - classified_observations (dict): A dictionary where each key is a stratum (as a string) and each value is a list of observations classified under that stratum.

    Returns:
    - list: A list representing the sample, where each element is a sublist representing an observation.
    """
    stratified_sample = []

    # Iterate over each stratum and replicate observations according to the sample size
    for key, sample_size in n_stratum.items():
        # Find observations classified under the current stratum
        observations = classified_observations[key]

        # Replicate the observations and extend the sample, taking into account the sample size
        stratified_sample.extend(observations[:sample_size])

    return stratified_sample


    return stratified_sample

def count_combinations_final(observations): 
    """
    Count the occurrences of each combination, ignoring the first element of each sublist.

    Args:
    - observations (list): A list of observations where each observation is a list.

    Returns:
    - dict: A dictionary where each key is a combination (as a tuple) and each value is the count of occurrences.
    """
    combinations_count = Counter()

    for obs in observations:
        # Ignore the first element (name) and count the rest as a tuple
        combination = tuple(obs[1:])
        combinations_count[combination] += 1

    return combinations_count
