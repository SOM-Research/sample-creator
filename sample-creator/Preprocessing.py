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


# FUNCIONES PRIVADAS
def read_dataframe(file_path): #En la que me he descargado yo no funciona, para comprobar lo he cargado de otra forma.
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
def dictionary_to_lists(dictionary): #creo que podriamos quitarlo. Esperar hasta ultima comprovacion para borrar
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

#Solo para categoricas
def count_elements_in_variables(variables_dict): # Descomentar el salto del name cuando probemos el csv bueno.
    """
    Count the occurrences of elements in each list of a variables dictionary, excluding the first variable because is the name, and return a dictionary of Counters.

    Args:
    - variables_dict (dict): The dictionary of variables where keys are variable names and values are lists of elements. The first variable is excluded from counting.

    Returns:
    - dict: A dictionary containing Counters for each variable's list of elements.
    """
    counters_dict = {}
    #first = True
    for variable, values in variables_dict.items():
        #if first:
            #first = False
            #continue
        counters_dict[variable] = Counter(values)
    return counters_dict
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


# MAIN FUCTION
def preprocessing(df):
    """
    Preprocesses a CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - dict: A dictionary containing numerical variables.
    - dict: A dictionary containing categorical variables.
    """
    # Read the CSV file
    #df = pd.read_csv(file_path)

    # Insert a new index - Solo era para mi caso, luego borrar!
    #df.insert(0, 'New_Index', range(1, len(df) + 1)) 
    #print(df) 

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

    return numerical_variables, categorical_variables



file_path = r"C:\Users\goros\OneDrive\Escritorio\UOC\type_likes.csv" 
df = pd.read_csv(file_path)
df.insert(0, 'New_Index', range(1, len(df) + 1)) #Esto solo es para los csv que tengo yo. Borrar desupes.
print(df) 
preprocessing(df)

'''
# MAIN CODE
df_clean = analyze_df(df)
variables = create_variables_dict_from_df(df_clean)

numerical_variables, categorical_variables = separate_numerical_categorical(variables)
# Mostramos las variables numéricas
print("Variables numéricas:")
for variable_name, values in numerical_variables.items():
    print(f"{variable_name}: {len(values)}")
# Mostramos las variables categóricas
print("\nVariables categóricas:")
for variable_name, values in categorical_variables.items():
    print(f"{variable_name}: {len(values)}")

# Para las variables categoricas
counters = count_elements_in_variables(categorical_variables)
print("Counter: ", counters)

keys, values = dictionary_to_lists(variables)
print("keys: ", keys, "values: ", len(values))
all_keys, all_values = dictionary_to_all_lists(counters)
print("All keys: ", all_keys)
print("All values: ", len(all_values))

statistics = print_and_collect_statistics(variables)
'''