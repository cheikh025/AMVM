"""Elite Guided Local Search
"""
import math

import numpy as np

from alns.stop import *

# ALNS used for intensification
from ALNS.ALNS import ALNS  # Import ALNS class
from Solution import Solution
from ALNS.State import State
from ALNS.local_search import LocalSearch

import utils.utils as ut

DIVERSIFY_RATE = 0.15  # Percentage of the array to diversify
# Set seed for reproducability
SEED = 9101
np.random.seed(SEED)



def solve(initial_row: np.ndarray, original_row: np.ndarray, inputs: np.ndarray,
            nQuantization: int, row_index=0, debug=False, main_loop_iterations=10, theta=10) -> Solution:
    """
    param: main_loop_iterations: number of outer loop iterations (i.e., number of diversifications done)
    param: theta: number of non-improvement iterations in the inner loop
    """

    current_row = initial_row  # Our initial S
    best_solution = None

    per_iteration_results = []

    # Calculate B_k first to save computation time
    B_k = np.matmul(original_row, inputs.T)

    # Main loop
    for iteration in range(main_loop_iterations):
        # Run our intensification algorithm, stopping when there is theta non-improving iterations

        # Setup the ALNS object
        intenseALNS = ALNS(current_row, original_row, inputs, nQuantization, debug)
        intenseALNS.set_stopping_criteria(NoImprovement(theta))  # Stop after theta iterations of noImprovement

        intenseALNS.row_index = 0
        intenseALNS.set_num_partials(100)  # |L.set| = 100

        solution = intenseALNS.solve()



        # Store the best so far
        if best_solution is None or solution.inf_norm_value < solution.best_state.objective():
            best_solution = solution

        # Store this iteration's results in an array
        per_iteration_results.append(solution.inf_norm_value)


        # Run diversification on the global best state

        # Set the weights of the best state to our current row
        # Note that we have to convert quantized values back to the integers
        current_row = solution.best_state.get_integer_weights(best_solution.quantized_weights)
        current_row = diversification(current_row, nQuantization)

        # TODO: Do restricted swap local search here
        # Perform swap local search
        # Make a temporary state
        temp_state = State(inputs, current_row, original_row, B_k, nQuantization, num_partial=100, debug=True)
        # local_search_swap(temp_state)

        # if temp_state.weights != current_row:
        #     print("something went wrong")




    if debug:
        # Print out the array with each iteration's values
        print(f"Best inf_norm for each iteration: {per_iteration_results}")


    method_name = "GuidedLS"
    best_solution.method_name = method_name
    return best_solution


def diversification(current_row: np.ndarray, nQuantization: int) -> np.ndarray:
    # TODO diversify on the global best
    """Diversifies the current row"""
    amount_to_destroy = int(len(current_row) * DIVERSIFY_RATE)
    indices_to_destroy = np.random.choice(np.arange(len(current_row)), size=amount_to_destroy)

    # Randomize the row
    for index in indices_to_destroy:
        current_row[index] = np.random.randint(low=0, high=2**nQuantization)

    return current_row

