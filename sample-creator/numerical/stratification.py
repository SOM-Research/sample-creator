import numpy as np
from sklearn.cluster import KMeans


def elbow_method(values, max_k=5): # FALTA POR IMPLEMENTAR. REVISAR SI ESTA IMPLEMENTADO
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

#One Numerical:
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