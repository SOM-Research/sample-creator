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


# STRATIFICATION
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
def get_stratum_dict(variables):
    """
    Get the stratum dictionary for the first variable.

    Args:
    - variables (dict): A dictionary where keys are variable names and values are lists of variable values.

    Returns:
    - dict: The stratum dictionary for the first variable.
    """
    # Obtain the dictionary of strata for the first variable
    stratum_dict = next(iter(variables.values()))
    
    return stratum_dict

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
def nis_phi(stratum_dict, N):
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
    nis = [len(obs_list) for obs_list in stratum_dict.values()]

    # Calculate the proportion of each stratum with respect to the total population
    phi = [ni / N for ni in nis]

    return nis, phi
def calculate_std_devs_single(stratum_dict):
    """
    Calculate the standard deviations for each stratum in the stratum dictionary.

    Args:
    - stratum_dict (dict): A dictionary containing strata for the variable.
                           The structure is {stratum_id: [values]}.

    Returns:
    - list: A list containing the standard deviations for each stratum.
    """
    std_devs = []
    
    # Iterate over each stratum in the dictionary
    for values in stratum_dict.values():
        std_dev = np.std(values)
        std_devs.append(std_dev)
    
    return std_devs
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


# SAMPLING
def sampling(strata_dict, phi_list, nis_list, s_list, ni):
    """
    Calculate statistics for stratified sampling.

    Args:
    - strata_dict (dict): Dictionary of strata where each key is a stratum and each value is a list of observations.
    - phi_list (list): List of stratum proportions.
    - nis_list (list): List of observations in each stratum.
    - s_list (list): List of standard deviations for each variable in each stratum.
    - ni (list): Sample size for each stratum.

    Returns:
    - tuple: Sample mean, lower confidence interval, upper confidence interval.
    """
    # Vector where the estimated mean of each stratum will be stored
    mean_Strata = [] 
    # Vector where the estimated variance of each stratum will be stored
    s2_Strata = [] 
    
    # Iterate over each stratum
    for i, obs_list in strata_dict.items(): 
        sample = np.random.choice(obs_list, ni[i])
        mean_Strata.append(np.mean(sample))
        s2_Strata.append(np.var(sample))

    # Estimation of the sample mean
    sum_mean = [phi_list[i] * mean_Strata[i] for i in range(len(phi_list))]
    mean = np.sum(sum_mean)

    # Standard error of the mean
    sx2 = [(((phi_list[i] ** 2) * (s_list[i] ** 2)) / ni[i]) * (1 - (ni[i] / nis_list[i])) for i in range(len(phi_list))]
    sx = np.sqrt(np.sum(sx2))

    # 95% confidence interval
    za2 = scipy.stats.norm.ppf(0.975)
    lower_interval = mean - sx * za2
    upper_interval = mean + sx * za2

    return mean, sx, lower_interval, upper_interval



# MAIN CODE - PREPROCESSING
print("PREPROCESSING:")
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\repository2.csv"
df = read_dataframe(file_path)
df = df.drop(columns=['type'])
print(df)
df_clean = analyze_df(df)
variables = create_lists_from_df(df_clean) #diccionario de las variables -> lista de valores
keys, values = dictionary_to_lists(variables)
statistics = print_and_collect_statistics(variables)


# MAIN CODE - STRATIFICATION
# elbow_method(values, 5)
print("STRATIFICATION:")
num_clusters_list = [3] # Specify the number of clusters for each variable
strata = create_stratum_kmeans(variables, num_clusters_list) # Apply KMeans clustering to each variable
print_stratum_counts(strata) #Print the number of elements in each stratum for each variable
stratum_dict = get_stratum_dict(strata)
print_stratum_counts({"First Variable": stratum_dict})


# MAIN CODE - PRESAMPLING
print("PRE-SAMPLING")
N, mu = extract_population_size_and_means(statistics)
print("Population size (N):", N)
print("Means (mu):", mu)
nis, phi = nis_phi(stratum_dict, N) 
s = calculate_std_devs(stratum_dict) # Calculate the standard deviations for each stratum
print("nis:", len(nis), nis)
print("phi:", len(phi), phi)
print("s: ", s)

#mu_rounded = round(mu[0], 2)
epsilon = mu[0] * 0.1
#epsilon_rounded = round(epsilon, 3)
n, ni = nStratifiedSampling(epsilon, 0.95, phi, s, 2, N, nis)
print("Size of the global sample:", n)
print("Sample sizes for each stratum:", ni)


# MAIN CODE - SAMPLING
# Llamada a la función calculate_statistics
mean_estimate, sampling_error, lower_confidence_interval, upper_confidence_interval = sampling(
    stratum_dict,   # Diccionario de estratos
    phi,      # Lista de proporciones de estrato
    nis,      # Lista de observaciones en cada estrato
    s,        # Lista de desviaciones estándar para cada variable en cada estrato
    ni        # Tamaño de la muestra para cada estrato
)


print("-------------------------------------------------------------------------------------------------")
print("LIKES")
print("N: ", N, " --> n: ", n)
print("Mu: ", mu[0], " --> x-barra: ", mean_estimate)
print("Intervalo de confianza: (", lower_confidence_interval, upper_confidence_interval, ")")
