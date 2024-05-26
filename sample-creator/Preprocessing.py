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

#ESTA SIN ACABAR
def count_elements_in_lists(lists_of_values):
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


# Main Code
file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\repository.csv"
df = read_dataframe(file_path)
print(df)
df_clean = analyze_df(df)
variables = create_lists_from_df(df_clean) #diccionario de las variables -> lista de valores
keys, values = dictionary_to_lists(variables)
statistics = print_and_collect_statistics(variables)


