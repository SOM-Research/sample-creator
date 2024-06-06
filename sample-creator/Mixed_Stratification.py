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
matplotlib.use('TkAgg')  # Use this backend for displaying plots

# PREPROCESSING
def read_dataframe(file_path): #POR AHORA NO LO USAMOS
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
def separate_numerical_categorical(variables):
    """
    Separate numerical and categorical variables from a dictionary containing variables.

    Args:
    - variables (dict): A dictionary where keys are variable names and values are lists of variable values.

    Returns:
    - dict: A dictionary containing numerical variables.
    - dict: A dictionary containing categorical variables.
    """
    numerical_variables = {}
    categorical_variables = {}

    for variable_name, values in variables.items():
        # Check the type of the first value to determine if the variable is numerical or categorical
        if isinstance(values[0], (int, float)) and not all(math.isnan(value) or value in (0, 1) for value in values):
            numerical_variables[variable_name] = values
        else:
            categorical_variables[variable_name] = values

    return numerical_variables, categorical_variables
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

#STRATIFICATION
def create_strata_kmeans(variables, num_clusters_list):
    """
    Apply KMeans clustering to each numeric variable in a list of variables with variable number of clusters.

    Args:
    - variables (dict): A dictionary where keys are variable names and values are lists of variable values.
    - num_clusters_list (list): A list of integers specifying the number of clusters for each variable.

    Returns:
    - dict: A dictionary containing strata for each variable. Each variable's strata are represented as a list of lists where each inner list contains the variable values belonging to each cluster.
    """
    strata = {}
    
    for (variable_name, values_list), num_clusters in zip(variables.items(), num_clusters_list):
        # Check if all values are numeric
        if all(isinstance(value, (int, float)) for value in values_list):
            # Initialize KMeans with specified number of clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            
            # Fit KMeans to the data
            kmeans.fit(np.array(values_list).reshape(-1, 1))

            # Get cluster labels
            labels = kmeans.labels_

            # Create strata list for the variable
            variable_stratum = [[] for _ in range(num_clusters)]
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
                     The structure is {variable_name: [[stratum_1_values], [stratum_2_values], ...]}.
    """
    for variable_name, stratum_list in strata.items():
        print(f"Variable: {variable_name}")
        for i, stratum_values in enumerate(stratum_list):
            print(f"  Stratum {i + 1}: {len(stratum_values)} points")
        print()
def create_strata_categoricals(counters_dict):
    """
    Create strata for each variable.

    Args:
    - counters_dict (dict): The dictionary of counters where keys are variable names and values are Counter objects.
      This dictionary represents the different subgroups of each variable, where each Counter object counts the occurrences of each subgroup.

    Returns:
    - dict: A dictionary containing strata for each variable.
      This dictionary has the same keys as the input counters_dict, but the values are lists of lists representing the strata for each variable.
      Each inner list represents a stratum and contains the elements corresponding to that stratum.
    """
    # Initialize an empty dictionary to store the strata for each variable
    strata_dict = {}

    # Iterate over each variable and its corresponding Counter object in the counters_dict
    for i, (variable, counter) in enumerate(counters_dict.items()):
        # Initialize an empty list to store the strata for the current variable
        strata = []
        # Iterate over each value and its count in the Counter object
        for value, count in counter.items():
            # Create a sublist for the current value repeated 'count' times
            stratum = [value] * count
            # Append the sublist to the strata list
            strata.append(stratum) 
        # Assign the strata list to the current variable in the strata dictionary
        strata_dict[variable] = strata

    # Return the dictionary containing the strata for each variable
    return strata_dict
def merge_strata_dicts(dict1, dict2):
    """
    Merge two dictionaries containing strata for variables into one dictionary.

    Args:
    - dict1 (dict): The first dictionary containing strata for variables.
    - dict2 (dict): The second dictionary containing strata for variables.

    Returns:
    - dict: A dictionary containing merged strata for variables.
    """
    merged_dict = {}

    # Merge keys from both dictionaries
    merged_dict.update(dict1)
    merged_dict.update(dict2)

    return merged_dict

# COMBINATION
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
    for variable_name, stratum_list in strata.items():
        variable_ranges = []
        # Iterate over the strata of the current variable
        for stratum_values in stratum_list:
            # Calculate the minimum and maximum values of the current stratum
            min_value = min(stratum_values)
            max_value = max(stratum_values)
            # Add the range [min_value, max_value] to the variable's ranges list
            variable_ranges.append([min_value, max_value])
        
        # Store the ranges of the current variable in the main ranges dictionary
        ranges[variable_name] = variable_ranges

    return ranges
def combination(numerical_ranges, categorical_keys):
    """
    Generate all possible combinations of elements from numerical ranges and categorical keys.

    Args:
    - numerical_ranges (dict): A dictionary containing ranges for each numerical variable.
    - categorical_keys (list): A list of lists where each inner list represents the keys (categories) of a categorical variable.

    Returns:
    - list: A list containing all possible combinations of elements from the input ranges and keys.
    """
    # Extract the lists of ranges for each numerical variable
    numerical_ranges_list = list(numerical_ranges.values())

    # Generate all combinations of the numerical ranges and categorical keys
    combinations = [list(comb) for comb in product(*categorical_keys, *numerical_ranges_list)]

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
def classify_mixed_observations(observations, combination_strata):
    """
    Classify observations into strata based on provided combinations, ignoring the first variable (assumed to be the name).

    Args:
    - observations (list): A list of observations where each observation is a list.
    - combination_strata (list): A list of lists containing combinations of strata.
                                 Each inner list contains a variable name and its associated range.

    Returns:
    - dict: A dictionary where each key is a stratum and each value is a list of observations.
    """
    classified_observations = {str(stratum): [] for stratum in combination_strata}

    # Iterate over each observation
    for obs in observations:
        # Skip the first element (assumed to be the name)
        name = obs[0]
        data_values = obs[1:]

        # Check which stratum the observation belongs to
        for stratum in combination_strata:
            variable, range_values = stratum
            min_val, max_val = range_values

            if variable == data_values[0] and min_val <= data_values[1] <= max_val:
                classified_observations[str(stratum)].append(obs)
                break

    return classified_observations

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
def nis_phi(classified_observations, N):
    """
    Calculate the number of observations in each stratum and their proportions with respect to the total population.

    Args:
    - classified_observations (dict): A dictionary containing classified observations where each key is a stratum and each value is a list of observations.
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

# SAMPLING
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


#MAIN CODE - PREPROCESSING
print("PREPROCESSING:")
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\repository.csv"
df = read_dataframe(file_path)
print(df)

# Clean the DataFrame
df_clean = analyze_df(df)

# Create variables dictionary from DataFrame
variables = create_variables_dict_from_df(df_clean)
keys, values = dictionary_to_lists(variables) #Creo que al final no es necesario. Esperar a la ultima comprobacion para borrar.

# Separate numerical and categorical variables
numerical_variables, categorical_variables = separate_numerical_categorical(variables)
# Mostramos las variables numéricas
print("Variables numéricas:")
for variable_name, values in numerical_variables.items():
    print(f"{variable_name}: {len(values)}")
# Mostramos las variables categóricas
print("\nVariables categóricas:")
for variable_name, values in categorical_variables.items():
    print(f"{variable_name}: {len(values)}")

# Further processing steps for categorical variables
counters = count_elements_in_variables(categorical_variables)
print("Counter: ", counters)
all_keys, all_values = dictionary_to_all_lists(counters)
print("keys: ", keys, "values: ", len(values))
print("All keys: ", all_keys)
print("All values: ", len(all_values))

# Finally, we compute and print the statistical variables
statistics = print_and_collect_statistics(variables)

# STRATIFICATION
print("STRATIFICATION")
#Stratification for categorical variables
categorical_strata_dict = create_strata_categoricals(counters) 
# Print the lengths of each stratum and each sublist in each variable
print("STRATA DICT:")
for variable, strata in categorical_strata_dict.items():
    print(f"Strata for variable '{variable}': {len(strata)}")
    for i, sublist in enumerate(strata, start=1):
        print(f"Length of stratum {i}: {len(sublist)}")

# Stratification for numerical variables
num_clusters_list = [3] # Specify the number of clusters for each variable
numerical_strata_dict = create_strata_kmeans(numerical_variables, num_clusters_list) # Apply KMeans clustering to each variable
print_stratum_counts(numerical_strata_dict) #Print the number of elements in each stratum for each variable

# Ejemplo de llamada a la función merge_strata_dicts
merged_strata = merge_strata_dicts(categorical_strata_dict, numerical_strata_dict)
print("MERGED STRATA")
# Imprimir la longitud de cada lista de valores en el diccionario fusionado
for variable_name, stratum_list in merged_strata.items():
    print(f"Variable: {variable_name}")
    for i, stratum_values in enumerate(stratum_list):
        print(f"  Stratum {i + 1}: {len(stratum_values)} points")
    print()



# MAIN CODE - COMBINATION
print("COMBINATION:")
numerical_ranges = get_stratum_ranges(numerical_strata_dict)
print("Ranges:", numerical_ranges)
combination_strata = combination(numerical_ranges, all_keys)
print(combination_strata)
print(len(combination_strata))
observations = df_to_list_observations(df_clean)

classified_observations = classify_mixed_observations(observations, combination_strata)
total_observations = 0
for stratum, obs_list in classified_observations.items():
    total_observations += len(obs_list)
    print(f"Stratum: {stratum}: {len(obs_list)} observations")
print("Total Observations:", total_observations)

# Verificar que cada sublista en las listas del diccionario tenga tres elementos
for comb, obs_list in classified_observations.items():
    for obs in obs_list:
        if len(obs) != 3:
            raise ValueError(f"Each sublist in the observations associated with combination {comb} must have a length of three.")
        

# MAIN CODE - PRESAMPLING
N, means = extract_population_size_and_means(statistics)
print(f"Population Size: {N}")
nis, phi = nis_phi(classified_observations, N)
print(f"Number of observations in each stratum (nis): {nis}")
print(f"Proportion of each stratum (phi): {phi}")

epsilon = 0.05
confidence = 0.95
n = sample_size(epsilon, confidence)
print("Required sample size:", n)

ni_size = determine_ni_size(phi, combination_strata, n)
print("SAMPLE SIZE OF EACH STRATUM")
for stratum_key, size in ni_size.items():
    print(f"Stratum {stratum_key}: {size} observations", type(size))



# MAIN CODE - SAMPLING
combined_sample = create_sample(classified_observations, ni_size)
print("Combined Sample Size:", len(combined_sample))

# Imprimir los primeros diez elementos del combined_sample
print("\nPrimeros diez elementos del combined sample:")
for i in range(min(10, len(combined_sample))):
    print(combined_sample[i])