"""Contains all repair operators"""
import math

from ALNS.local_search import *

from ALNS.operator_utils import *




# REPAIR OPERATORS -------------------------------------------------------------------------------
def random_repair(state: State, rnd_state: np.random.RandomState) -> State:

    # Operator debug output
    operator_debug(state, random_repair)
    old_state_l2_norm = state.L2_norm.clone()
    old_objective = state.objective_value

    removed_indices = [i for i in range(len(state.removed_array))
                       if state.removed_array[i] and not state.fixed_mask[i]]
    changed_indices = []
    changed_values = []
    for index in removed_indices:
        rand = rnd_state.randint(low=-1, high=2)
        candidate = int(state.weights[index]) + rand
        if rand != 0 and 0 <= candidate < state.num_levels:
            changed_indices.append(index)
            changed_values.append(candidate)
    if changed_indices:
        state.apply_move(changed_indices, changed_values)

    # Run local search before returning
    ls = LocalSearch(state)
    state = ls.run()

    if not state.accepts(state.objective_value, state.L2_norm,
                         incumbent_linf=old_objective, incumbent_l2=old_state_l2_norm):
        state.objective_value = math.inf

    return state

def greedy_repair(state: State, rnd_state: np.random.RandomState) -> State:
    """
        Try to increase state.weights[j] by 1 step or decrease by 1 step
        Then take the best change out of these 2, or do no change if they only make the solution worse
    """

    operator_debug(state, greedy_repair)  # Operator debug output

    removed_indices = [i for i in range(len(state.removed_array))
                       if state.removed_array[i] and not state.fixed_mask[i]]

    rnd_state.shuffle(removed_indices)
    for j in removed_indices:
        current = int(state.weights[j])
        candidates = [q for q in (current - 1, current + 1) if 0 <= q < state.num_levels]
        admissible = []
        for candidate in candidates:
            residual = state.move_residual([j], [candidate])
            linf = residual.abs().max().item()
            l2 = residual.square().sum()
            if state.accepts(linf, l2):
                admissible.append((linf, l2.item(), candidate))
        if admissible:
            _, _, best = min(admissible)
            state.apply_move([j], [best])
    # Run local search after the repair operator
    ls = LocalSearch(state)
    state = ls.run()
    return state
