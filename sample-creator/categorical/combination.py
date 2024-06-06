from itertools import product


def combination(all_keys): # ONLY CAT
    """
    Generate all possible combinations of elements from strata.

    Args:
    - strata (list): A list of lists, where each inner list represents the strata of a variable to be combined.

    Returns:
    - list: A list containing all possible combinations of elements from the input strata.
    """
    return [list(comb) for comb in product(*all_keys)]

def df_to_list_observations(df): # NOT REQUIRED, PUT CODE INTO FUNCTION
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

def count_combinations(observations, combination):
    """
    Counts the occurrences of each specific combination of strata in the observations.

    Parameters:
    observations (list): List of observations, where each observation is a list containing the name,
                         the stratum type, and the stratum value.
    combination (list): List of possible combinations of strata, where each combination is a tuple 
                        containing the stratum type and the stratum value.

    Returns:
    dict: A dictionary where the keys are the combinations of strata (format "(type, value)")
          and the values are the counts of how many times each combination appears in the observations.
    """
    # Create the initial dictionary with all possible combinations and an initial count of 0
    combination_strata = {f"({comb[0]}, {comb[1]})": 0 for comb in combination}

    # Iterate over the observations to count the specific combinations
    for obs in observations:
        # Ignore the first value (name) of each observation
        obs_combination = obs[1:]  # Take the elements after the first one
        for comb in combination:
            # Check if the observation matches the current combination
            if obs_combination == list(comb):
                key = f"({comb[0]}, {comb[1]})"
                combination_strata[key] += 1

    return combination_strata

def classify_observations(observations, combination_strata):
    """
    Classify observations into strata based on provided combinations, ignoring the first variable (assumed to be the name).

    Args:
    - observations (list): A list of observations where each observation is a list.
    - combination_strata (list): A list of lists containing combinations.

    Returns:
    - dict: A dictionary where each key is a stratum (as a string) and each value is a list of observations.
    """
    # Initialize the dictionary with the combination keys
    classified_observations = {}

    # Iterate over each observation
    for obs in observations:
        # Extract the variables (excluding the name)
        obs_variables = obs[1:]
        # Iterate over each combination to classify the observation
        for comb in combination_strata:
            if obs_variables == comb:
                key = f"({comb[0]}, {comb[1]})"
                if key not in classified_observations:
                    classified_observations[key] = []
                classified_observations[key].append(obs)
                break  # Stop checking once the observation is classified

    return classified_observations
