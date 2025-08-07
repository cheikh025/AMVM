"""Functions or constants used by operators"""

import numpy as np
import sys
from ALNS.State import State

# CONSTANTS FOR REMOVE/REPAIR OPERATORS
DESTROY_RATE = 0.005  # Percentage of the array to destroy
ALPHA_COEFFICIENT = 0.3  # Coefficient a for exp{-a * R_jk}, for worst_remove

def remove_operator_debug(state: State) -> None:
    """Prints debug output if state.debug is set to true, also updates the iteration"""
    # Debugging output
    state.iteration += 1  # Add 1 to the current iteration
    if state.debug:
        print(f"Iteration {state.iteration} -------------")

def operator_debug(state: State, func) -> None:
    """Prints debug output for all remove/repair operators, if debug is true"""
    if state.debug:
        print(f"Using operator {func.__name__}")
