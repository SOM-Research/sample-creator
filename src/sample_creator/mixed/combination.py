from itertools import product


def get_stratum_ranges(strata): # SAME AS NUM.
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

def combination(numerical_ranges, categorical_keys): # ONLY FOR MIX.
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

def df_to_list_observations(df): # CAN GO TO CLASSIFY 
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

def classify_mixed_observations(observations, combination_strata): # ONLY FOR MIX
    """
    Classify observations into strata based on provided combinations, ignoring the first variable (assumed to be the name).

    Args:
    - observations (list): A list of observations where each observation is a list.
    - combination_strata (list): A list of lists containing combinations of strata.
                                 Each inner list contains a variable name and its associated range.

    Returns:
    - dict: A dictionary where each key is a stratum (as a string) and each value is a list of observations.
    """
    classified_observations = {str(stratum): [] for stratum in combination_strata}

    # Iterate over each observation
    for data_values in observations:
        # # Skip the first element (assumed to be the name)
        # name = obs[0]
        # data_values = obs[1:]

        # Check which stratum the observation belongs to
        for stratum in combination_strata:
            variable, range_values = stratum
            min_val, max_val = range_values

            if variable == data_values[0] and min_val <= data_values[1] <= max_val:
                classified_observations[str(stratum)].append(data_values)
                break

    return classified_observations

