from itertools import product


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
    empty_strata = []  

    for stratum in strata_combination:
        obs_list = classified_observations.get(str(stratum), [])  
        num_observations = len(obs_list)  
        if num_observations == 0:
            empty_strata.append(stratum)  

    for stratum in empty_strata:
        del classified_observations[str(stratum)]

def print_combination_stratum_counts(classified_observations, strata_combinations):
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
        obs_list = classified_observations.get(str(stratum), [])  
        num_observations = len(obs_list)  
        if num_observations > 0:  
            print(f"Stratum {stratum}: {num_observations} observations")
        total_observations += num_observations

    print(f"\nTotal sum of observations in all strata: {total_observations}")