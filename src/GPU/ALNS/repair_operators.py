"""Contains all repair operators"""
import math

from ALNS.local_search import *

from ALNS.operator_utils import *




# REPAIR OPERATORS -------------------------------------------------------------------------------
def random_repair(state: State, rnd_state: np.random.RandomState) -> State:

    # Operator debug output
    operator_debug(state, random_repair)
    old_state_l2_norm = state.L2_norm

    # Get indices of removed elements
    removed_indices = [i for i in range(len(state.removed_array)) if state.removed_array[i]]
    # For each index, change it to some random index in Q
    for index in removed_indices:
        # state.weights[index] = rnd_state.randint(low=0, high=state.num_levels)
        # Testing: Only change one at max
        rand = rnd_state.randint(low=-1, high=2)
        # rand = rnd_state.randint(low=-state.weights[index], high=state.num_levels-state.weights[index])
        # Additional check that index is not an outlier
        if not state.outlier_mask[index] and rand != 0 and 0 <= state.weights[index] + rand < state.num_levels:
            state.weights[index] += rand

            # Add these changes to state
            state.changes.append((index, rand))

    # Perform changes to the objective function
    state.eval_flag = FULL
    state.recalculate_flag = True
    state.objective()

    # Run local search before returning
    ls = LocalSearch(state)
    state = ls.run()
    state.eval_flag = FULL
    state.recalculate_flag = True

    # L2NORM: comment this out to accept worse L2 norms
    if state.L2_norm >= old_state_l2_norm:
        # If L2 is too big after this, then don't accept it
        state.eval_flag = FULL
        state.recalculate_flag = True
        state.objective()
        state.objective_value = math.inf

    return state

def greedy_repair(state: State, rnd_state: np.random.RandomState) -> State:
    """
        Try to increase state.weights[j] by 1 step or decrease by 1 step
        Then take the best change out of these 2, or do no change if they only make the solution worse
    """

    operator_debug(state, greedy_repair)  # Operator debug output

    removed_indices = [i for i in range(len(state.removed_array)) if state.removed_array[i]]

    rnd_state.shuffle(removed_indices)
    # Unpack the new L_set tuple
    D_ks, row_indices = state.L_set  # D_all: full signed-D tensor, row_indices: which rows we're tracking
    state.objective()
    for j in removed_indices:
        curr_objective = state.objective_value #state.objective()
        #print('Curr ', curr_objective)

        # If we can, evaluate the result if we decrease/increase W_j by 1 (i.e. if state.weights[j] != 0)
        decrease_eval = math.inf  # Store the result if we decrease W_j here
        if state.weights[j] > 0:
            state.change = (-1, j)  # Change W_j by -1 step
            state.eval_flag = PARTIAL_D_SINGLE_CHANGE  # Evaluate but don't apply change
            state.recalculate_flag = False
            decrease_eval = state.objective()
            #print("down ", decrease_eval)

        increase_eval = math.inf
        if state.weights[j] + 1 < state.num_levels:
            state.change = (1, j)
            state.eval_flag = PARTIAL_D_SINGLE_CHANGE
            state.recalculate_flag = False 
            increase_eval = state.objective()
            #print("UP ", increase_eval)

        changed_delta_Wj = 0
        if decrease_eval > curr_objective and increase_eval > curr_objective:
            state.change = (0, j)
            continue  # Don't make changes since none of them decrease the objective
        elif decrease_eval < increase_eval:
            # Then we want to decrease W_j by 1 step
            changed_delta_Wj = -1
        else:
            changed_delta_Wj = 1

        if changed_delta_Wj != 0:  # We have a proposed change
          #  # Temporarily store the best objective
            state.objective_value = min(decrease_eval, increase_eval)
            state.change = (changed_delta_Wj, j)

            # Apply the weight change
            state.weights[j] += changed_delta_Wj
            state.changes.append((j, changed_delta_Wj))  # Store these changes for objective recalculation
    #print("=================================")
    state.recalculate_flag = True
    state.eval_flag       = FULL
    state.objective()
    # Run local search after the repair operator
    ls = LocalSearch(state)
    state = ls.run()
    state.recalculate_flag = True
    state.eval_flag       = FULL
    state.objective()
    return state
