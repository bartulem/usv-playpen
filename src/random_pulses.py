"""
@author: bartulem
Generates truly random seeds.
"""

import json
import quantumrandom


def generate_truly_random_seed(input_parameter_dict=None):
    """
    Description
    ----------
    This method generates a truly random seed. The process is based on the
    ANU Quantum Random Number Generator (see: https://qrng.anu.edu.au/),
    generated in real-time in the lab by measuring the quantum fluctuations
    of the vacuum.
    ----------

    Parameters
    ----------
    Contains the following set of parameters
        dtype (str)
            The data type, can be 'uint16' or 'hex16'; defaults to 'uint16'.
        array_len (int)
            The number of random seeds to generate; defaults to 1.
    ----------

    Returns
    ----------
    quantum_seed_values (list)
        A list of random seeds.
    ----------
    """

    # load .json parameters
    if input_parameter_dict is None:
        with open('input_parameters.json', 'r') as json_file:
            input_parameter_dict = json.load(json_file)['random_pulses']['generate_truly_random_seed']

    quantum_seed_values = quantumrandom.get_data(data_type=input_parameter_dict['dtype'],
                                                 array_length=input_parameter_dict['array_len'])

    return quantum_seed_values
