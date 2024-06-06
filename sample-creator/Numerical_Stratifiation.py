# We import all the necessary librarys
import numpy as np
import random
import math
import scipy.stats as stats
from sklearn.cluster import KMeans
from scipy.stats import skew
import scipy
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib
from itertools import product

# PREPROCESSING
def read_dataframe(file_path):
    """
    Reads a DataFrame from a CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The DataFrame read from the CSV file.
    """
    try:
        df = pd.read_csv(file_path, header=0)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
def analyze_df(df):
    """
    Analyze a DataFrame, remove rows with NaN values in numerical columns, and print the number of NaN values.

    Args:
    - df (DataFrame): The input DataFrame.

    Returns:
    - DataFrame: The cleaned DataFrame with rows removed if NaN values exist in numerical columns.
    """
    if df.isnull().values.any():
        # Print the number of NaN values
        print("Number of NaN values: \n", df.isnull().sum())
        
        # Identify numerical and categorical columns
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Remove rows with NaN values in numerical columns
        df_clean = df.dropna(subset=numerical_columns)
        
        return df_clean
    else:
        print("The dataframe does not contain NaN values.")
    
    # Return the original DataFrame if no NaN values exist
    return df
def create_lists_from_df(df, columns=None):
    """
    Create lists of column values from a DataFrame.

    Args:
    - df (DataFrame): The input DataFrame.
    - columns (list): Optional list of columns to include. Default is None (all columns).

    Returns:
    - dict: A dictionary containing column names as keys and lists of column values as values. 
    """
    column_lists = {}
    
    if columns is None:
        columns = df.columns
    
    for column_name in columns:
        column_values = df[column_name].tolist()
        column_lists[column_name] = column_values
    
    return column_lists
def count_elements_in_lists(lists_of_values): #NO SE SI ES NECESARIO
    """
    Count elements in lists of values.

    Args:
    - lists_of_values (list): A list of lists of values.

    Returns:
    - dict: A dictionary where keys are lists and values are counts of elements.
    """
    distributions = {}
    for values_list in lists_of_values:
        counter = Counter(values_list)
        distributions[tuple(values_list)] = dict(counter)
    return distributions
def dictionary_to_lists(dictionary):
    """
    Convert a dictionary into separate lists for keys and values.

    Args:
    - dictionary (dict): The input dictionary.

    Returns:
    - tuple: A tuple containing two lists: one for keys and one for values.
    """
    # Extract keys and values from the dictionary
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    
    # Return the lists
    return keys, values
def print_and_collect_statistics(variables):
    """
    Print statistics for each variable in the dictionary and collect them in a dictionary.

    Args:
    - variables (dict): A dictionary where keys are variable names and values are lists of variable values.

    Returns:
    - dict: A dictionary containing the statistics for each variable.
    """
    # Dictionary to store the statistics for each variable
    stats_dict = {}
    
    # Iterate over each variable in the dictionary, but ignore the first variable
    for i, (variable_name, values_list) in enumerate(variables.items()):
        if i == 0:
            continue  # Skip the first variable
        
        # Check if the variable is numerical (assuming the first value represents the type)
        if isinstance(values_list[0], (int, float)):
            # Calculate statistics for numerical variables
            N = len(values_list)
            mean = np.mean(values_list)
            std_dev = np.std(values_list)
            median = np.median(values_list)
            
            # Store statistics in a dictionary
            stats = {
                'Population size': N,
                'Mean': mean,
                'Median': median,
                'Standard Deviation': std_dev
            }
            
            # Print statistics
            print(f"Statistics for numerical variable '{variable_name}':")
            print(f"  Population size: {N}")
            print(f"  Mean: {mean}")
            print(f"  Median: {median}")
            print(f"  Standard Deviation: {std_dev}\n")
        else:
            # Calculate the number of observations for categorical variables
            N = len(values_list)
            
            # Store the number of observations in a dictionary
            stats = {
                'Number of observations': N
            }
            
            # Print the number of observations
            print(f"Statistics for categorical variable '{variable_name}':")
            print(f"  Number of observations: {N}\n")
        
        # Add the statistics of the current variable to the main dictionary
        stats_dict[variable_name] = stats
    
    # Return the dictionary with all the statistics
    return stats_dict

#STRATIFICATION for Numerical Variables
def elbow_method(values, max_k=5):
    return None
def create_stratum_kmeans(variables, num_clusters_list):
    """
    Apply KMeans clustering to each numeric variable in a list of variables with variable number of clusters.

    Args:
    - variables (dict): A dictionary where keys are variable names and values are lists of variable values.
    - num_clusters_list (list): A list of integers specifying the number of clusters for each variable.

    Returns:
    - dict: A dictionary containing strata for each variable. Each variable's strata are represented as a dictionary where keys are cluster labels and values are lists of variable values belonging to each cluster.
    """
    strata = {}
    
    # Skip the first variable and iterate over the remaining variables and their corresponding number of clusters
    for (i, (variable_name, values_list)) in enumerate(zip(variables.keys(), variables.values())):
        if i == 0:
            continue  # Skip the first variable
        
        num_clusters = num_clusters_list[i - 1]  # Adjust index for num_clusters_list

        # Check if all values are numeric
        if all(isinstance(value, (int, float)) for value in values_list):
            # Initialize KMeans with specified number of clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            
            # Fit KMeans to the data
            kmeans.fit(np.array(values_list).reshape(-1, 1))

            # Get cluster labels
            labels = kmeans.labels_

            # Create strata dictionary for the variable
            variable_stratum = {i: [] for i in range(num_clusters)}
            for j, data in enumerate(values_list):
                variable_stratum[labels[j]].append(data)

            # Store strata for the variable in the overall strata dictionary
            strata[variable_name] = variable_stratum
        else:
            print(f"Skipping non-numeric variable '{variable_name}'.")

    return strata
def print_stratum_counts(strata):
    """
    Print the number of elements in each stratum for each variable.

    Args:
    - strata (dict): A dictionary containing strata for each variable.
                     The structure is {variable_name: {stratum_id: [values]}}.
    """
    for variable_name, stratum_dict in strata.items():
        print(f"Variable: {variable_name}")
        for stratum_id, elements in stratum_dict.items():
            print(f"  Stratum {stratum_id + 1}: {len(elements)} points")
        print()

#COMBINATION
def get_stratum_ranges(strata):
    """
    Get the ranges of each stratum for each variable.

    Args:
    - strata (dict): A dictionary containing strata for each variable.

    Returns:
    - dict: A dictionary with the ranges for each stratum of each variable.
    """
    ranges = {}

    # Iterate over each variable in the strata dictionary
    for variable_name, variable_stratum in strata.items():
        variable_ranges = []
        # Iterate over the strata of the current variable
        for stratum, values in variable_stratum.items():
            # Calculate the minimum and maximum values of the current stratum
            min_value = min(values)
            max_value = max(values)
            # Add the range [min_value, max_value] to the variable's ranges list
            variable_ranges.append([min_value, max_value])
        
        # Store the ranges of the current variable in the main ranges dictionary
        ranges[variable_name] = variable_ranges

    return ranges
def combination(ranges):
    """
    Generate all combinations of the ranges to create combination strata.

    Args:
    - ranges (dict): A dictionary containing ranges for each variable.

    Returns:
    - list: A list of lists where each sublist is a combination of range elements from the variables.
    """
    # Extract the lists of ranges for each variable
    strata_lists = list(ranges.values())
    
    # Generate all combinations of the ranges
    combinations = [list(comb) for comb in product(*strata_lists)]
    
    return combinations
def df_to_list_observations(df):
    """
    Convert the rows of a DataFrame into a list of lists.

    Args:
    - df (pd.DataFrame): The DataFrame to convert.

    Returns:
    - list: A list of lists where each sublist is a row from the DataFrame.
    """
    # Get the rows of the DataFrame as a list of lists
    list_of_lists = df.values.tolist()
    return list_of_lists
def classify_observations(observations, strata_ranges):
    
    """
    Classify observations into strata based on provided ranges, ignoring the first variable (assumed to be the name).

    Args:
    - observations (list): A list of observations where each observation is a list.
    - strata_ranges (list): A list of lists containing ranges for each variable.

    Returns:
    - dict: A dictionary where each key is a stratum (as a string) and each value is a list of observations.
    """
    classified_observations = {str(stratum): [] for stratum in strata_ranges}

    # Iterate over each observation
    for obs in observations:
        # Skip the first element (assumed to be the name)
        name = obs[0]
        data_values = obs[1:]

        # Check which stratum the observation belongs to
        for i, stratum in enumerate(strata_ranges):
            in_range = True
            for j, (min_val, max_val) in enumerate(stratum):
                if not (min_val <= data_values[j] <= max_val):
                    in_range = False
                    break
            
            if in_range:
                classified_observations[str(stratum)].append(obs)
                break

    return classified_observations
def drop_empty_strata(classified_observations, strata_combination):
    """
    Drop stratum with 0 observations from the classified observations dictionary.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.
    - strata_combination (list): A list containing the combination of strata to consider.

    Returns:
    - None
    """
    empty_strata = []  # Lista para almacenar los estratos con 0 observaciones
    for stratum in strata_combination:
        obs_list = classified_observations.get(str(stratum), [])  # Obtener la lista de observaciones para el estrato actual
        num_observations = len(obs_list)  # Obtener el número de observaciones en el estrato
        if num_observations == 0:
            empty_strata.append(stratum)  # Agregar el estrato a la lista de estratos vacíos

    # Eliminar los estratos con 0 observaciones del diccionario de observaciones clasificadas
    for stratum in empty_strata:
        del classified_observations[str(stratum)]
def print_combination_stratum_counts(classified_observations, strata_combinations):
    """
    Print the number of observations in each stratum in the order of defined strata.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.
    - strata_combinations (list): A list containing the combination of strata to consider.

    Returns:
    - None
    """
    total_observations = 0
    for stratum in strata_combinations:
        obs_list = classified_observations.get(str(stratum), [])  # Obtener la lista de observaciones para el estrato actual
        num_observations = len(obs_list)  # Obtener el número de observaciones en el estrato
        if num_observations > 0:  # Solo imprimir si hay observaciones en el estrato
            print(f"Stratum {stratum}: {num_observations} observations")
        total_observations += num_observations

    print(f"\nTotal sum of observations in all strata: {total_observations}")

#PRE-SAMPLING
def extract_population_size_and_means(statistics):
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
            N = stats['Population size']  # Assuming the population size is the same for all variables
        mu.append(stats['Mean'])
    
    return N, mu
def nis_phi(classified_observations, N):
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
    Calculate the sample size for stratified sampling based on given parameters.

    Args:
    - epsilon (float): Desired precision.
    - confidence (float): Confidence level (e.g., 0.95 for 95% confidence).
    - phi (list): Proportion of each stratum with respect to the total population.
    - s (list): Standard deviation of each stratum.
    - setting (int): Type of setting for sampling fraction calculation:
        - 1: Equal allocation to each stratum.
        - 2: Proportional allocation based on stratum proportion.
        - 3: Optimal allocation based on he variability in each stratum.
    - N (int): Total population size.
    - nis (list): Number of observations in each stratum.

    Returns:
    - tuple: A tuple containing:
        - n (int): Total sample size.
        - n_strata (list): Sample size of each stratum.
    """   
    K = len(phi)  # Number of strata
    alfa = 1 - confidence
    za = scipy.stats.norm.ppf(1 - alfa / 2)

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
def split_list(input_list): #CREO QUE AL FINAL NO LA USE
    # Unpack the input list and use zip to transpose rows to columns
    transposed = list(zip(*input_list))

    # Convert tuples to lists (optional, but often more convenient)
    output_lists = [list(x) for x in transposed]

    return output_lists
def calculate_sample_sizes(mu, confidence, phi, all_s, setting, N, nis):
    """
    Calculate the sample sizes for each variable calling to the fuction nStratifiedSampling.

    Args:
    - mu (list): List of means for numerical variables.
    - confidence (float): Confidence level (e.g., 0.95 for 95% confidence).
    - phi (list): Proportion of each stratum with respect to the total population.
    - all_s (list): List of standard deviations for each stratum of each variable.
    - setting (int): Type of setting for sampling fraction calculation:
        - 1: Equal allocation to each stratum.
        - 2: Proportional allocation based on stratum proportion.
        - 3: Optimal allocation based on he variability in each stratum.
    - N (int): Total population size.
    - nis (list): List of number of observations in each stratum for each variable.

    Returns:
    - tuple: A tuple containing:
        - sample_sizes (list): Sample size for each variable.
        - strata (list): Sample size of each stratum for each variable.
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


# SAMPLING
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
    - tuple: Estimated mean, lower confidence interval, upper confidence interval.
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
    za2 = scipy.stats.norm.ppf(0.975)
    lower_interval = mean - sx * za2
    upper_interval = mean + sx * za2

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
    variable_names = list(variables.keys()) 
    
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

# MAIN CODE - PREPROCESSING
print("PREPROCESSING:")
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\likes_downloads.csv"
df = read_dataframe(file_path)
print(df)
df_clean = analyze_df(df)
variables = create_lists_from_df(df_clean) #diccionario de las variables -> lista de valores
keys, values = dictionary_to_lists(variables)
statistics = print_and_collect_statistics(variables)

# MAIN CODE - STRATIFICATION
# elbow_method(values, 5)
print("STRATIFICATION:")
num_clusters_list = [3, 3] # Specify the number of clusters for each variable
strata = create_stratum_kmeans(variables, num_clusters_list) # Apply KMeans clustering to each variable
print_stratum_counts(strata) #Print the number of elements in each stratum for each variable


# MAIN CODE - COMBINATION
print("COMBINATION:")
stratum_ranges = get_stratum_ranges(strata)
print("Ranges:", stratum_ranges)
strata_combinations = combination(stratum_ranges) 
print("Combinations of ranges:") 
for comb in strata_combinations:
    print(comb)
print(len(strata_combinations))
observations = df_to_list_observations(df_clean)
print(len(observations))

classified_observations_before = classify_observations(observations, strata_combinations)

# Print the number of observations in each stratum in the order of defined strata
print("\nBefore dropping empty strata:")
print_combination_stratum_counts(classified_observations_before, strata_combinations)
# Make a copy of classified observations
classified_observations_after = classified_observations_before.copy()
# Drop stratum with 0 observations
drop_empty_strata(classified_observations_after, strata_combinations)
# Print the number of observations in each stratum in the order of defined strata after dropping empty strata
print("\nAfter dropping empty strata:")
print_combination_stratum_counts(classified_observations_after, strata_combinations)

# MAIN CODE - PRESAMPLING
print("PRE-SAMPLING")
N, mu = extract_population_size_and_means(statistics)
print("Population size (N):", N)
print("Means (mu):", mu)
nis, phi = nis_phi(classified_observations_after, N) # Calculate the stratum proportions
s = calculate_variable_std_devs(classified_observations_after) # Calculate the standard deviations for each stratum
print("nis:", len(nis), nis)
print("phi:", len(phi), phi)
print("s: ", s)
print("s:")
for variable_index, std_devs_for_variable in enumerate(s, start=1):
    print(f"  Variable {variable_index}: {std_devs_for_variable}")

sample_sizes, strata = calculate_sample_sizes(mu, 0.95, phi, s, 2, N, nis) # Call the function to calculate sample sizes
print("Size of the global sample:", sample_sizes)
print("Sample sizes for each stratum:", strata)

max_n, max_n_dist, max_n_idx = get_max_sample_distribution(sample_sizes, strata)
print("Variable wiht Max n: ", max_n, max_n_dist)

filtered_max_n_dist, filtered_phi, filtered_nis, filtered_s, filtered_classified_observations = filter_zero_strata(
    max_n_dist=max_n_dist,
    phi=phi,
    nis=nis,
    s=s,
    max_n_idx=max_n_idx,
    classified_observations=classified_observations_after
)

print("Filtered max_n_dist:", len(filtered_max_n_dist), filtered_max_n_dist)
print("Filtered phi:", len(filtered_phi), filtered_phi)
print("Filtered nis:", len(filtered_nis), filtered_nis)
print("Filtered s:", len(filtered_s), filtered_s)
print("Filtered classified_observations:", len(filtered_classified_observations))

# MAIN - SAMPLING 
automated_sampling(filtered_classified_observations, filtered_phi, filtered_nis, filtered_s, max_n_idx, max_n, filtered_max_n_dist)

