# We import all the necessary librarys
import numpy as np
import random
import math
import scipy.stats as stats
from sklearn.cluster import KMeans
from scipy.stats import skew
from scipy.stats import norm
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
    - file_path (str): The path to the CSV file. The CSV file is expected to have the following format:
      - The first value represents the name.
      - The following columns represent the variables.

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
    Analyze a DataFrame, removed NaN values are from numerical columns but retained in categorical columns. Print the number of NaN values.

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

    """
    Analyze a DataFrame, removed NaN values are from numerical columns but retained in categorical columns. Print the number of NaN values.

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
def create_variables_dict_from_df(df, columns=None): 
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
def count_elements_in_variables(variables_dict): 
    """
    Count the occurrences of elements in each list of a variables dictionary, excluding the first variable because is the name, and return a dictionary of Counters.

    Args:
    - variables_dict (dict): The dictionary of variables where keys are variable names and values are lists of elements. The first variable is excluded from counting.

    Returns:
    - dict: A dictionary containing Counters for each variable's list of elements.
    """
    counters_dict = {}
    first = True
    for variable, values in variables_dict.items():
        if first:
            first = False
            continue
        counters_dict[variable] = Counter(values)
    return counters_dict
def dictionary_to_lists(dictionary): #creo que este se usa en numericas y el siguiente categoricas. JOIN
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
def dictionary_to_all_lists(dictionary): 
    """
    Convert a dictionary into separate lists of unique keys for each variable.

    Args:
    - dictionary (dict): The input dictionary.

    Returns:
    - tuple: A tuple containing two lists of lists: one for keys and one for values.
    """
    # Initialize empty lists for keys and values
    all_keys = []
    all_values = []
    
    # Extract keys and values from each variable's dictionary
    for variable_dict in dictionary.values():
        keys = list(variable_dict.keys())
        values = list(variable_dict.values())
        all_keys.append(keys)
        all_values.append(values)
    
    return all_keys, all_values
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
        
        # Calculate the population size for all variables
        N = len(values_list)
        
        # Check if the variable is numerical (assuming the first value represents the type)
        if isinstance(values_list[0], (int, float)):
            # Calculate statistics for numerical variables
            mean = np.mean(values_list)
            std_dev = np.std(values_list)
            median = np.median(values_list)
            
            # Store statistics in a dictionary
            stats = {
                'Population Size': N,
                'Mean': mean,
                'Median': median,
                'Standard Deviation': std_dev
            }
            
            # Print statistics
            print(f"Statistics for numerical variable '{variable_name}':")
            print(f"  Population Size: {N}")
            print(f"  Mean: {mean}")
            print(f"  Median: {median}")
            print(f"  Standard Deviation: {std_dev}\n")
        else:
            # Store the number of observations for categorical variables
            stats = {
                'Population Size': N
            }
            
            # Print the number of observations
            print(f"Statistics for categorical variable '{variable_name}':")
            print(f"  Population Size: {N}\n")
        
        # Add the statistics of the current variable to the main dictionary
        stats_dict[variable_name] = stats
    
    # Return the dictionary with all the statistics
    return stats_dict

# STRATIFICATION for Categorical Variables
def create_strata(counters_dict):
    """
    Create strata based on the counts from a dictionary of counters.

    Args:
    - counters_dict (dict): A dictionary of counters where keys are variable names and values are Counter objects.
      This dictionary represents the different subgroups of each variable, where each Counter object counts the occurrences of each subgroup.

    Returns:
    - dict: A dictionary containing strata for each variable.
      This dictionary has keys as the elements in the Counter objects from counters_dict values, and the values are lists representing the strata for each variable.
      Each list contains elements repeated according to their counts in the corresponding Counter object.
    """
    # Initialize an empty dictionary to store the strata for each variable
    strata_dict = {}

    # Iterate over each variable and its corresponding Counter object in the counters_dict
    for variable, counter in counters_dict.items():
        # Iterate over each element in the Counter
        for element, count in counter.items():
            # Create a list repeating the element count times
            stratum = [element] * count
            # Add the list to the strata dictionary using the element as key
            if element in strata_dict:
                strata_dict[element].extend(stratum)
            else:
                strata_dict[element] = stratum

    # Return the dictionary containing the strata for each variable
    return strata_dict


# PRE-SAMPLING
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
def nis_phi(strata_dict, N): 
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
    nis = [len(obs_list) for obs_list in strata_dict.values()]

    # Calculate the proportion of each stratum with respect to the total population
    phi = [ni / N for ni in nis]

    return nis, phi
def sample_size(epsilon, confidence):
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
def determine_ni_size(phi, all_keys, n):
    """
    Calculate the sample size for each stratum based on proportions and the desired total sample size.

    Args:
    - phi (list): A list containing the proportion of each stratum with respect to the total population.
    - all_keys (list): A list containing the unique keys for the variable.
    - n (int): The desired total sample size.

    Returns:
    - dict: A dictionary where each key is a stratum (as a string) and each value is the calculated sample size for that stratum.
    """
    # Number of strata
    K = len(phi)

    # Initialize the dictionary to store the sample size for each stratum
    n_stratum = {}
    total_allocated = 0

    # Calculate the initial sample size for each stratum
    for i, proportion in enumerate(phi):
        if all_keys:
            stratum_key = all_keys[0][i]  # Extract the stratum key from the first (and only) list in all_keys
        else:
            # Handle the case when all_keys is empty
            # For example, you might want to raise an error or return early
            # depending on the desired behavior of your function
            return {}

        ni = round(proportion * n)  # Calculate the sample size for the current stratum
        n_stratum[stratum_key] = ni  # Store the sample size in the dictionary
        total_allocated += ni

    # Calculate the difference between the total allocated and the desired total sample size
    difference = n - total_allocated

    # Distribute the difference proportionally among the strata
    if difference != 0:
        for i, proportion in enumerate(phi):
            if all_keys:
                stratum_key = all_keys[0][i]  # Extract the stratum key from the first (and only) list in all_keys
            else:
                # Handle the case when all_keys is empty
                # For example, you might want to raise an error or return early
                # depending on the desired behavior of your function
                return {}

            additional_allocation = round(proportion * difference)
            n_stratum[stratum_key] += additional_allocation

    return n_stratum


# SAMPLING
def create_sample(n_stratum, strata_dict):
    """
    Create a sample based on the provided sample sizes for each stratum and the strata dictionary.

    Args:
    - n_stratum (dict): A dictionary where each key is a stratum (as a string) and each value is the calculated sample size for that stratum.
    - strata_dict (dict): A dictionary where each key is a stratum (as a string) and each value is a list of observations classified under that stratum.

    Returns:
    - list: A list representing the sample, where each element is a sublist representing an observation.
    """
    stratified_sample = []

    # Iterate over each stratum and replicate observations according to the sample size
    for key, sample_size in n_stratum.items():
        # Find observations classified under the current stratum
        observations = strata_dict[key]

        # Replicate the observations and extend the sample, taking into account the sample size
        stratified_sample.extend(observations[:sample_size])

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

# MAIN CODE - PREPROCESSING
print("PREPROCESSING:")
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\repository2.csv" # Read the CSV file without an index
df = pd.read_csv(file_path)
df = df.drop(columns=['likes'])
print(df)
df_clean = analyze_df(df)
variables = create_variables_dict_from_df(df_clean) 
counters = count_elements_in_variables(variables)
print(counters)

keys, values = dictionary_to_lists(variables)
print("keys: ", keys, "values: ", len(values))
all_keys, all_values = dictionary_to_all_lists(counters)
print("All keys: ", all_keys)
print("All values: ", len(all_values))

statistics = print_and_collect_statistics(variables)


# MAIN CODE - STRATIFICATION Categoricals
print("STRATIFICATION")
# Call the create_strata function to create the strata
strata_dict = create_strata(counters) 
print("STRATA DICT:")
for key, values in strata_dict.items():
    print(f"Longitud de la lista de valores para '{key}': {len(values)}")

# MAIN CODE - PRESAMPLING
print("PRE-SAMPLING")
N, means = extract_population_size_and_means(statistics)
print(f"Population Size: {N}")
nis, phi = nis_phi(strata_dict, N)
print(f"Number of observations in each stratum (nis): {nis}")
print(f"Proportion of each stratum (phi): {phi}")

epsilon = 0.05
confidence = 0.95
n = sample_size(epsilon, confidence)
print("Required sample size:", n)

ni_size = determine_ni_size(phi, all_keys, n)
print("SAMPLE SIZE OF EACH STRATUM")
for stratum_key, size in ni_size.items():
    print(f"Stratum {stratum_key}: {size} observations")

# MAIN CODE - SAMPLING
# Create the stratified sample
sample = create_sample(ni_size, strata_dict)

# Print the stratified sample size and a few sample entries to verify
print(f"Total stratified sample size: {len(sample)}")
print("Sample entries:")
for i in range(10):  # Print the first 10 sample entries
    print(sample[i])


# Llamada a la función para contar las combinaciones
final_combination_counts = count_combinations_final(sample)

# Imprimir el resultado
print("Counts of each combination:")
for combination, count in final_combination_counts.items():
    print(f"Combination: {combination}, Count: {count}")
