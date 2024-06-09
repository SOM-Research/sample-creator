import math
from scipy.stats import norm
import random


def extract_population_size_and_means(statistics): # ALL
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

def nis_phi(classified_observations, N): # ALL
    """
    Calculate the number of observations in each stratum and their proportions with respect to the total population.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum
                                       and each value is a list of observations.
    - N (int): The total population size.

    Returns:
    - tuple: A tuple containing two lists:
             1. A list containing the number of observations in each stratum.
             2. A list containing the proportion of each stratum with respect to the total population.
    """
    # Calculate the number of observations in each stratum
    nis = [len(obs_list) for obs_list in classified_observations.values()]

    # Calculate the proportion of each stratum with respect to the total population
    phi = [ni / N for ni in nis]

    return nis, phi

def sample_size(epsilon, confidence): # CAT
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

def determine_ni_size(phi, combination_strata, n): # ONLY FOR MIX. MAYBE SAME AS CAT, TO REVIEW.
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

def create_sample(classified_observations, ni_size):
    """
    Create a combined sample from classified observations based on the sample sizes determined for each stratum.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.
    - ni_size (dict): A dictionary where each key is a stratum (as a string) and each value is the calculated sample size for that stratum.

    Returns:
    - list: A list containing the combined sample of observations.
    """
    sample = []

    # Iterate over the values of both dictionaries simultaneously
    for (classified_obs_list, n_samples) in zip(classified_observations.values(), ni_size.values()):
        # If the sample size for the current stratum is zero, skip to the next stratum
        if n_samples == 0:
            continue
        
        # If the sample size is greater than the number of observations in the stratum,
        # add all observations in the stratum to the sample
        if n_samples >= len(classified_obs_list):
            sample.extend(classified_obs_list)
        else:
            # Otherwise, randomly select n_samples observations from the stratum and add them to the sample
            sample.extend(random.sample(classified_obs_list, n_samples))

    return sample

def create_sample_old(n_stratum, classified_observations): # TODO: STILL WORKING...
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
        print(f"Stratum: {key}, Sample size: {sample_size}")  # Print stratum and sample size
        # Find observations classified under the current stratum
        observations = classified_observations.get(key, [])
        
        # Check if sample size is greater than 0 and there are observations available
        if sample_size > 0 and observations:
            print(f"Taking {sample_size} observations from {len(observations)} available")  # Print number of observations to take
            # Take a random sample of observations of the specified size
            sampled_observations = random.sample(observations, min(sample_size, len(observations)))
            # Extend the stratified sample with the sampled observations
            stratified_sample.extend(sampled_observations)

    return stratified_sample

