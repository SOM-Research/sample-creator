import numpy as np
import math
import pandas as pd
from collections import Counter

def read_dataframe(csv_file_path):
    """
    Reads a DataFrame from a CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: The DataFrame read from the CSV file.
    """
    try:
        df = pd.read_csv(csv_file_path, header=0)
        return df
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None

def remove_nan_df(df): # Old analyze_df
    """
    Remove rows with NaN values in numerical columns, and print the number of NaN values.

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
        
        # Remove rows with NaN values in numerical columns
        df.dropna(subset=numerical_columns, inplace = True)
    else:
        print("The dataframe does not contain NaN values.")

def create_lists_from_df(df, columns=None): # SAME AS "create_variables_dict_from_df"
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

