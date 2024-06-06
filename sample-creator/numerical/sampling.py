import numpy as np
import scipy.stats as stats

def extract_population_size_and_means(statistics): # SAME FOR CAT, NUM, MIXT
    """
    Extract the population size and means from the statistics dictionary.

    Args:
    - statistics (dict): A dictionary containing statistical information for each variable.

    Returns:
    - tuple: A tuple containing the population size (N) and a list of means (mu) for each variable.
    """
    N = None
    mu = []
    
    for variable_name, stats in statistics.items():
        if N is None:
            N = stats['Population Size']  # Assuming the population size is the same for all variables # TODO: That's not always true
        if 'Mean' in stats:
            mu.append(stats['Mean'])
    
    return N, mu

def nis_phi(classified_observations, N): # SAME FOR CAT, NUM, PHI
    """
    Calculate the number of observations in each stratum and their proportions with respect to the total population.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.

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

def calculate_variable_std_devs(classified_observations): 
    """
    Calculate the standard deviations of each variable for each stratum, excluding the first variable (assumed to be names).

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.

    Returns:
    - list: A list of lists containing the standard deviations of each variable for each stratum.
    """
    # List to store the standard deviations of each variable for each stratum
    std_devs_by_variable = []

    # Iterate over each variable (excluding the first variable assumed to be names)
    for variable_index in range(1, len(next(iter(classified_observations.values()))[0])):
        # List to store the standard deviations for the current variable across strata
        std_devs_for_variable = []

        # Iterate over each stratum
        for stratum_observations in classified_observations.values():
            # Get the values of the current variable in the current stratum
            variable_values = [float(obs[variable_index]) for obs in stratum_observations[1:] if isinstance(obs[variable_index], (int, float))]

            if variable_values:
                # Calculate the standard deviation of the values of the current variable in the current stratum
                std_dev = np.std(variable_values)
            else:
                std_dev = 0.0

            # Append the standard deviation to the result for the current variable
            std_devs_for_variable.append(std_dev)

        # Append the standard deviations of the current variable to the final result
        std_devs_by_variable.append(std_devs_for_variable)

    return std_devs_by_variable

def nStratifiedSampling(epsilon, confidence, phi, s, setting, N, nis):
    """
    JUNE: EXPLAIN FUNCTION
    """
    K = len(phi)  # Number of strata
    alfa = 1 - confidence
    za = stats.norm.ppf(1 - alfa / 2)

    # We calculate the sampling fraction for each stratum
    if setting == 1:
        w = np.repeat(1 / K, K)
    elif setting == 2:
        w = phi
    else:
        w = []
        sum_den = []
        for n, s_value in zip(nis, s):
            sum_den.append(n * s_value)

        for i in range(K):
            numerator = nis[i] * s[i]
            denominator = np.sum(sum_den)
            w.append(numerator / denominator)

    # We calculate the global size of the sample
    numerator = np.sum([((phi[i]**2) * s[i]**2) / w[i] for i in range(K)])
    denominator = (epsilon / za)**2 + 1 / N * np.sum([phi[i] * (s[i]**2) for i in range(K)])

    n = int(np.ceil(numerator / denominator))

    # From n, we calculate the sample size of each stratum (ni = wi * n)
    n_strata = [int(np.floor(n * w[i])) for i in range(K)]

    # Calculate the difference to distribute
    total_assigned = sum(n_strata)
    difference = n - total_assigned

    # Distribute the difference by incrementing some strata
    i = 0
    while difference > 0:
        n_strata[i] += 1
        difference -= 1
        i = (i + 1) % K

    return n, n_strata

def calculate_sample_sizes(mu, confidence, phi, all_s, setting, N, nis):
    """
    JUNE: TO FILL
    """
    sample_sizes = []
    strata = []
    
    # Iterate over each variable and calculate epsilon and call nStratifiedSampling
    for variable_mu, s_for_variable, nis_for_variable in zip(mu, all_s, nis):
        epsilon = variable_mu * 0.1  # Calculate epsilon for the current variable
        n, ni = nStratifiedSampling(epsilon, confidence, phi, s_for_variable, setting, N, nis_for_variable)
        sample_sizes.append(n)
        strata.append(ni)

    return sample_sizes, strata

def get_max_sample_distribution(sample_sizes, strata):
    """
    Get the maximum sample distribution and its corresponding sample size.

    Args:
    - sample_sizes (list): List containing the sample sizes for each stratum.
    - strata (list): List containing the sample distributions for each stratum.

    Returns:
    - tuple: Maximum sample size and its corresponding sample distribution.
    """
    # Take the var with max value
    max_n = max(sample_sizes)
    max_n_idx = sample_sizes.index(max_n)
    max_n_dist = strata[max_n_idx]
    return max_n, max_n_dist, max_n_idx

def filter_zero_strata(max_n_dist, phi, nis, s, max_n_idx, classified_observations):
    """
    Filter out strata with zero observations from the input lists and dictionary.

    Args:
    - max_n_dist (list): List representing the distribution of the stratum with the maximum sample size.
    - phi (list): List of stratum proportions.
    - nis (list): List of number of observations in each stratum.
    - s (list of lists): List of lists of standard deviations for each stratum and variable.
    - max_n_idx (int): Index of the stratum with the maximum sample size.
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.

    Returns:
    - tuple: A tuple containing filtered max_n_dist, phi, nis, s lists, and a filtered classified_observations dictionary.
    """
    indices_to_remove = [i for i, x in enumerate(max_n_dist) if x == 0]

    # Filter out elements with zero observations
    filtered_max_n_dist = [x for i, x in enumerate(max_n_dist) if i not in indices_to_remove]
    filtered_phi = [x for i, x in enumerate(phi) if i not in indices_to_remove]
    filtered_nis = [x for i, x in enumerate(nis) if i not in indices_to_remove]
    filtered_s = [[x for i, x in enumerate(s_var) if i not in indices_to_remove] for s_var in s]

    # Remove corresponding items from classified_observations
    items = list(classified_observations.items())
    filtered_classified_observations = {items[i][0]: items[i][1] for i in range(len(items)) if i not in indices_to_remove}

    return filtered_max_n_dist, filtered_phi, filtered_nis, filtered_s, filtered_classified_observations

def sampling(classified_observations, index, phi_list, nis_list, s_list, max_n_idx, max_n, max_n_dist):
    """
    Calculate sampling statistics.

    Args:
    - classified_observations (dict): Dictionary of classified observations.
    - index (int): Index of the variable to sample.
    - phi_list (list): List of stratum proportions.
    - nis_list (list): List of observations in each stratum.
    - s_list (list): List of standard deviations for each variable in each stratum.
    - max_n_idx (int): Index of the variable with the maximum sample size.
    - max_n (int): Maximum sample size.
    - max_n_dist (list): Distribution of the maximum sample size across strata.

    Returns: 
    - tuple: Estimated mean, sampling error, lower confidence interval, upper confidence interval.
    """
    mean_Strata = []  # Vector where the estimated mean of each stratum will be stored
    s2_Strata = []    # Vector where the estimated variance of each stratum will be stored
    
    # Iterate over each stratum
    for i, obs in enumerate(classified_observations.values()): 
        var_obs = [x[index] for x in obs] 
        sample = np.random.choice(var_obs, max_n_dist[i])
        mean_Strata.append(np.mean(sample))
        s2_Strata.append(np.var(sample))

    # Estimation of the sample mean
    sum_mean = [phi_list[i] * mean_Strata[i] for i in range(len(phi_list))]
    mean = np.sum(sum_mean)

    # Standard error of the mean
    sx2 = [(((phi_list[i] ** 2) * (s_list[index-1][i] ** 2)) / max_n_dist[i]) * (1 - (max_n_dist[i] / nis_list[i])) for i in range(len(phi_list))]
    sx = np.sqrt(np.sum(sx2))

    # 95% confidence interval
    za2 = stats.norm.ppf(0.975)
    lower_interval = mean - sx * za2
    upper_interval = mean + sx * za2

    # Sampling error
    #error = sampling_error(0.95, max_n)

    return mean, lower_interval, upper_interval 

def automated_sampling(classified_observations, phi, nis, s, max_n_idx, max_n, max_n_dist):
    """
    Perform sampling for all variables in classified_observations and print the results.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.
    - phi (list): List of stratum proportions. 
    - nis (list): List of number of observations in each stratum.
    - s (list): List of standard deviations for each stratum.
    - max_n_idx (int): Index of the stratum with the maximum sample size.
    - max_n (int): Maximum sample size.
    - max_n_dist (list): List representing the distribution of the stratum with the maximum sample size.
    """
    variable_names = list(variables.keys()) # TODO: Where does this variable come from? Which variable should be?
    
    for idx, variable_name in enumerate(variable_names): 
        if idx == 0:
            continue 

        mean, lower_interval, upper_interval = sampling(
            classified_observations=classified_observations,
            index=idx,   
            phi_list=phi,
            nis_list=nis,
            s_list=s, 
            max_n_idx=max_n_idx,
            max_n=max_n,
            max_n_dist=max_n_dist
        )

        # Print the results
        print(f"Results for variable '{variable_name}':")
        print("  Total sample size: ", max_n)
        print("  Estimated mean: ", mean)
        print("  95% confidence interval: (", lower_interval, ",", upper_interval, ")")
        print("----------------------------------------------------------------")
