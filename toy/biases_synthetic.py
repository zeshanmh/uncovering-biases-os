''' 
Selection bias: 
-  Non-random selection of participants into the study leads to a 
non-representative sample.
'''
import logging
import numpy as np

def apply_selection_bias(data, selection_dependence='Y'):
    """
    Applies selection bias by selecting samples based on specified dependence.
    
    Parameters:
    - data: DataFrame, observational study data
    - selection_dependence: str, 'Y', 'A', or 'both' 
        - 'Y': selection probability increases with higher Y values
        - 'A': selection probability increases with higher A values
        - 'both': selection probability depends on both A and Y
    
    Returns:
    - DataFrame after applying selection bias
    """
    if selection_dependence == 'Y':
        # Higher probability to include based on outcome Y
        selection_prob = 0.7 * (data['Y'] > data['Y'].median()) + 0.3
    elif selection_dependence == 'A':
        # Higher probability to include based on exposure A
        selection_prob = 0.6 * data['A'] + 0.4
    elif selection_dependence == 'both':
        # Dependence on both A and Y
        selection_prob = 0.5 * data['A'] + 0.5 * (data['Y'] > data['Y'].median())
    else:
        raise ValueError("Unsupported selection_dependence. Choose 'Y', 'A', or 'both'.")

    S = np.random.binomial(1, selection_prob, len(data))
    data_selection_bias = data[S == 1].copy()
    return data_selection_bias

def apply_bias_wrapper(data, bias_dict):
    """
    Applies a type of bias (based on type given in bias_dict) 
    to the observational data.
    
    Parameters:
    - data: DataFrame, observational study data
    - bias_dict: dict, dictionary containing the type of bias to apply 
    and the parameters of the bias.
    
    Returns:
    - DataFrame after applying bias
    """ 
    bias_type = bias_dict["type"]

    if bias_type == "selection_bias":
        logging.info("Bias applied successfully")
        return apply_selection_bias(data, bias_dict["params"]["selection_dependence"])
    else:
        logging.warning(f"No bias applied")
        return data
