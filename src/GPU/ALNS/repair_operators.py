"""Contains all repair operators"""
import math

from ALNS.local_search import *

from ALNS.operator_utils import *




def repairable_indices(state: State) -> list:
    """Return the removed, non-fixed variable indices in increasing order.

    One device reduction instead of one host synchronization per variable. The
    per-element form this replaced cost 113 ms per call at M=768 on MPS against
    1 ms here, and it scales with M, so the gap grows on the wide layers.
    """
    removed = torch.as_tensor(state.removed_array, device=state.torch_device,
                              dtype=torch.bool)
    return torch.nonzero(removed & ~state.fixed_mask, as_tuple=True)[0].tolist()


def current_levels_of(state: State, indices: list) -> list:
    """Return ``state.weights`` at ``indices`` as host integers in one transfer."""
    if not indices:
        return []
    return state.weights[torch.as_tensor(indices, device=state.torch_device,
                                         dtype=torch.long)].tolist()


# REPAIR OPERATORS -------------------------------------------------------------------------------
def random_repair(state: State, rnd_state: np.random.RandomState) -> State:

    # Operator debug output
    operator_debug(state, random_repair)
    old_state_l2_norm = state.L2_norm.clone()
    old_objective = state.objective_value

    removed_indices = repairable_indices(state)
    # Gather the current levels once. The random draws stay in the same order and
    # the same count, so the search trajectory is unchanged.
    current_levels = current_levels_of(state, removed_indices)
    changed_indices = []
    changed_values = []
    for position, index in enumerate(removed_indices):
        rand = rnd_state.randint(low=-1, high=2)
        candidate = current_levels[position] + rand
        if rand != 0 and 0 <= candidate < state.num_levels:
            changed_indices.append(index)
            changed_values.append(candidate)
    if changed_indices:
        state.apply_move(changed_indices, changed_values)

    # Run local search before returning
    state = run_local_search(state)

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

    removed_indices = repairable_indices(state)

    rnd_state.shuffle(removed_indices)
    # Each index is visited once and a move touches only that index, so the
    # levels gathered here stay current for the variable being repaired.
    current_levels = dict(zip(removed_indices, current_levels_of(state, removed_indices)))
    for j in removed_indices:
        current = current_levels[j]
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
    state = run_local_search(state)
    return state
