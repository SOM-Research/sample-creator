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
def read_dataframe(file_path): #POR AHORA NO LO USAMOS
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
    Count the occurrences of elements in each list of a variables dictionary, excluding the first variable, 
    and return a dictionary of Counters.

    Args:
    - variables_dict (dict): The dictionary of variables where keys are variable names and values are lists of elements.

    Returns:
    - dict: A dictionary containing Counters for each variable's list of elements, excluding the first variable.
    """
    counters_dict = {}
    first = True
    for variable, values in variables_dict.items():
        if first:
            first = False
            continue
        counters_dict[variable] = Counter(values)
    return counters_dict

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
        
        # Check if the variable is numerical or boolean (assuming the first value represents the type)
        if isinstance(values_list[0], (int, float)) and set(values_list) != {0, 1}:
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
            if isinstance(values_list[0], bool):
                print(f"Statistics for boolean variable '{variable_name}':")
            else:
                print(f"Statistics for categorical variable '{variable_name}':")
            print(f"  Number of observations: {N}\n")
        
        # Add the statistics of the current variable to the main dictionary
        stats_dict[variable_name] = stats
        
    return stats_dict
def plot_pie_charts(keys, values): #FALTA POR ARREGLAR
    """
    Plot pie charts for each variable using keys and values.

    Args:
    - keys (list): A list of variable names.
    - values (list): A list of lists of variable values.
    """
    num_variables = len(keys)

    # We create the subfigures
    fig, axes = plt.subplots(1, num_variables, figsize=(15, 7))

    for i, (variable_name, variable_values) in enumerate(zip(keys, values)):
        # Plot pie chart
        axes[i].pie(variable_values, labels=variable_values, autopct='%1.1f%%')
        axes[i].set_title(f'Pie Chart for {variable_name}')

    plt.show()
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




# STRATIFICATION for Numerical Variables
def create_strata(counters_dict):
    """
    Create strata for variables.

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
        # Skip the first variable
        if i == 0:
            continue

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

# COMBINATION
def combination(strata):
  # Usamos itertools.product para generar todas las combinaciones entre los elementos de las listas de entrada.
  # Calculamos el producto cartesiano de las listas usando itertools.product y luego convertimos cada tupla en una lista.
  return [list(comb) for comb in product(*strata)]
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
def create_combination_strata(combinations, observations):
    """
    Create a dictionary to count the occurrences of each combination of variable subgroups.

    Args:
    - combinations (list): A list of tuples where each tuple is a combination of variable values.
    - observations (list): A list of observations where each observation is a tuple of variable values.

    Returns:
    - dict: A dictionary with keys as combinations of variable subgroups and values as counts of those combinations.
    """
    # Initialize the combination_strata dictionary with keys from combinations and values set to 0
    combination_strata = {"(" + ", ".join(map(str, comb)) + ")": 0 for comb in combinations}
    
    # Count the occurrences of each combination in the observations
    for obs in observations:
        for comb in combinations:
            if tuple(obs) == comb:
                key = f"({', '.join(map(str, comb))})"
                combination_strata[key] += 1

    return combination_strata

# MAIN CODE - PREPROCESSING
print("PREPROCESSING:")
'''
file_path = "C:/Users/goros/OneDrive/Escritorio/UOC/space.csv"
df = read_dataframe(file_path)
print(df)
'''
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\type_pull.csv" # Read the CSV file without an index
df = pd.read_csv(file_path)
df.insert(0, 'New_Index', range(1, len(df) + 1)) # Add a new column as index
print(df)# Display the DataFrame with the new index
df_clean = analyze_df(df)
variables = create_variables_dict_from_df(df_clean) #diccionario de las variables -> lista de valores
counters = count_elements_in_variables(variables)
print(counters)

keys, values = dictionary_to_lists(variables)
all_keys, all_values = dictionary_to_all_lists(counters)
print("All keys: ", all_keys)
print("All values: ", len(all_values))

statistics = print_and_collect_statistics(variables)
#plot_pie_charts(keys, values)



# MAIN CODE - STRATIFICATION Categoricals
print("STRATIFICATION")
# Call the create_strata function to create the strata
strata_dict = create_strata(counters) 
# Print the lengths of each stratum and each sublist in each variable
for variable, strata in strata_dict.items():
    print(f"Strata for variable '{variable}': {len(strata)}")
    for i, sublist in enumerate(strata, start=1):
        print(f"Length of stratum {i}: {len(sublist)}")



# MAIN CODE - COMBINATION
print("COMBINATION")
combination_strata = combination(all_keys)
print(combination)
observations = df_to_list_observations(df_clean)
print(len(observations))

# Create combination strata
combination_strata = create_combination_strata(combination_strata, observations)

# Print the result
print(combination_strata)
