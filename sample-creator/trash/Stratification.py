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


# Stratification for Numerical Variables
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
    
    # Hay que pensar como hacer lo del nombre! Si es df mixto no llegara con nombre aqui, pero si es solo numerico si.
    for (i, (variable_name, values_list)) in enumerate(zip(variables.keys(), variables.values())):
        num_clusters = num_clusters_list[i - 1]  # Adjust index for num_clusters_list

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

# Stratification for Categorical Variables
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

# In case we have mixed data, union of the two dictionarys 
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