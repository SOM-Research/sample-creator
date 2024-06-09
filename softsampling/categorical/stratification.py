
def create_strata_single(counters_dict):
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
    for variable, counter in counters_dict.items():
        # Iterate over each element in the Counter
        for element, count in counter.items():
            # Create a list repeating the element count times
            stratum = [element] * count
            # Add the list to the strata dictionary using the element as key
            if element in strata_dict:
                strata_dict[element].extend(stratum)
            else:
                strata_dict[element] = stratum

    # Return the dictionary containing the strata for each variable
    return strata_dict

def create_strata_multiple(counters_dict):
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
## OLD CODE: TODO (delete)
    # # Initialize an empty dictionary to store the strata for each variable
    # strata_dict = {}

    # # Iterate over each variable and its corresponding Counter object in the counters_dict
    # for i, (variable, counter) in enumerate(counters_dict.items()):
    #     # Initialize an empty list to store the strata for the current variable
    #     strata = []
    #     # Iterate over each value and its count in the Counter object
    #     for value, count in counter.items():
    #         # Create a sublist for the current value repeated 'count' times
    #         stratum = [value] * count
    #         # Append the sublist to the strata list
    #         strata.append(stratum) 
    #     # Assign the strata list to the current variable in the strata dictionary
    #     strata_dict[variable] = strata

    # # Return the dictionary containing the strata for each variable
    # return strata_dict