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
matplotlib.use('TkAgg')  # Use this backend for displaying plots

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
def create_lists_from_df(df):
    """
    Create lists of column values from a DataFrame.

    Args:
    - df (DataFrame): The input DataFrame.

    Returns:
    - dict: A dictionary containing column names as keys and lists of column values as values.
    """
    # Dictionary to store lists of column values
    column_lists = {}
    
    # Iterate over each column in the DataFrame
    for column_name in df.columns:
        # Convert column values to a list and store in the dictionary
        column_values = df[column_name].tolist()
        column_lists[column_name] = column_values
    
    # Return the dictionary
    return column_lists
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
def count_elements_in_lists(lists_of_values): #ESTA SIN HACER
    distributions = {}
    for values_list in lists_of_values[1:]:
        counter = Counter(values_list)
        distributions[tuple(values_list)] = dict(counter)
    return distributions
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
    - dict: A dictionary containing strata for each variable.
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
def print_stratum_counts(classified_observations, strata_combinations):
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
def nis_phi(variables, N):
    return None
def s(variables):
    return None
def nStratifiedSampling(epsilon, confidence, phi, s, setting, N, nis):
    return None

# Main Code - Preprocessing
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\likes_downloads.csv"
df = read_dataframe(file_path)
print(df)
df_clean = analyze_df(df)
variables = create_lists_from_df(df_clean) #diccionario de las variables -> lista de valores
keys, values = dictionary_to_lists(variables)
statistics = print_and_collect_statistics(variables)

# Main code - Stratification
# elbow_method(values, 5)
num_clusters_list = [3, 3] # Specify the number of clusters for each variable
strata = create_stratum_kmeans(variables, num_clusters_list) # Apply KMeans clustering to each variable

# Main code - Combination
stratum_ranges = get_stratum_ranges(strata)
print(stratum_ranges)
strata_combinations = combination(stratum_ranges) 
print("Combinations of ranges:") 
for comb in strata_combinations:
    print(comb)
print(len(strata_combinations))
observations = df_to_list_observations(df_clean)
print(len(observations))

classified_observations = classify_observations(observations, strata_combinations)

# Print the number of observations in each stratum in the order of defined strata
print_stratum_counts(classified_observations, strata_combinations)

# Drop stratum with 0 observations
drop_empty_strata(classified_observations, strata_combinations)

# Print the number of observations in each stratum in the order of defined strata after dropping empty strata
print_stratum_counts(classified_observations, strata_combinations)