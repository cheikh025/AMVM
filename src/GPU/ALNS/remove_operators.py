"""Remove operators and their helper functions"""

import copy
from .State import *
import math

from ALNS.operator_utils import *

def create_copied_state(state: State, removed_array: torch.tensor):


    new_weights_array = torch.clone(state.weights)  # Make a new weights row too for modification
    newState = copy.copy(state)  # Make a copy of the state to modify it
    newState.removed_array = removed_array  # Store the indicated elements to remove here
    newState.weights = new_weights_array
    newState.eval_flag = FULL

    # Make copies of L_set and full eval
    L_set_vals = torch.clone(state.L_set[0])
    L_set_ind = torch.clone(state.L_set[1])

    newState.L_set = (L_set_vals, L_set_ind)

    newState.signedD_ks = torch.clone(state.signedD_ks)
    newState.absD_ks = torch.clone(state.absD_ks)
    # newState.L_set = state.L_set.copy(deep=True)
    # newState.full_eval = state.full_eval.copy(deep=True)

    # Set RECALCULATE_FLAG to true (to ensure we recalculate it after repairing)
    newState.recalculate_flag = True

    return newState

# Remove Operators:
def random_remove(state: State, rnd_state: np.random.RandomState) -> State:
    remove_operator_debug(state)  # Prints debug output if debug is true
    operator_debug(state, random_remove)

    M = len(state.weights)  # Get the size of the row (first dimension)
    num_to_remove = int(DESTROY_RATE * M)  # Number of elements to remove
    to_remove = rnd_state.choice(a=M, size=num_to_remove)

    removed_array = torch.zeros(len(state.weights), dtype=torch.bool, device=state.torch_device)  # Make a copy of the removed array
    removed_array[to_remove] = True  # Set selected removed elements to 1

    return create_copied_state(state, removed_array)

def worst_remove(state: State, rnd_state: np.random.RandomState) -> State:
    """
    Remove weights based off range size (for now, we can change this to look for how close a weight w_j is to its bound)

    Parameters
    ----------
    state
    rnd_state
    Returns
    -------

    """

    # Operator debug output
    remove_operator_debug(state)
    operator_debug(state, worst_remove)

    removed_array = np.zeros(len(state.weights), dtype=bool)
    #removed_array = torch.zeros(len(state.weights), dtype=bool)

    # L_set is a tuple: (tensor of signed D_k values, tensor of row_index values)
    D_ks = state.L_set[0]
    indices = state.L_set[1]

    D_ks = D_ks.reshape((-1, 1))#.to(state.inputs.device)
    sum_abs_D_k = D_ks.abs().sum().item()

    t_value = torch.tensor(state.objective(), device=state.inputs.device)
    
    row_weights = state.quantization_levels[state.weights.reshape((1, -1))]#.to(state.inputs.device)  # [ w_1  w_2  ...  w_m  ]

    # Get the num_partials * m matrix of inputs based on the indices in L_set
    partial_inputs = state.inputs[indices]  # Get the partial inputs: this is matrix X of size num_partials * m

    # Compute the element-wise product of the broadcasted weight matrix and the partial input matrix
    element_wise_product = row_weights * partial_inputs
    # Our matrix now looks like this, with row weight vector broadcasted into k rows
    # [ w_1 * x_11  w_2 * x_12  ...  w_m * x_1m ]
    # [ ...                                ...  ]
    # [ w_1 * x_k1  w_2 * x_k2  ...  w_m * x_km ]

    # LB_jk = (w_jk - D_k - t) / x_kj
    # Prevent divide by 0 errors
    mask = partial_inputs != 0

    LB_matrix = torch.where(
        mask,
        torch.div(element_wise_product - (t_value + D_ks), partial_inputs),
        torch.zeros_like(element_wise_product)
    )
    UB_matrix = torch.where(
        mask,
        torch.div(element_wise_product - (-t_value + D_ks), partial_inputs),
        torch.zeros_like(element_wise_product)
    )

    LB_matrix -= row_weights
    UB_matrix -= row_weights

    LB_matrix = LB_matrix.abs()  # These matrices should now have the distance between w_j and its respective bound
    UB_matrix = UB_matrix.abs()

    dist_matrix = torch.minimum(LB_matrix, UB_matrix) * -ALPHA_COEFFICIENT   # Take the min distance and multiply it by -a

    individual_scores = torch.exp(dist_matrix).T  # transpose it to a m * k matrix so we can do matmul with a k * 1 D_k vector

    if sum_abs_D_k == 0:
        return state  # Prevent divide by 0 errors

    score_tensor = individual_scores.matmul(D_ks.abs()).div(sum_abs_D_k).flatten()
    score_tensor = score_tensor.div(score_tensor.sum())  # Normalize

    # Destroy destroy_rate percent of the array
    #amount_to_destroy = int(len(state.weights) * DESTROY_RATE)
    amount_to_destroy = math.ceil(len(state.weights) * DESTROY_RATE)
    removed_idx_tensor = torch.multinomial(score_tensor, amount_to_destroy, replacement=False)
    removed_indices = removed_idx_tensor.cpu().numpy()

    removed_array[removed_indices] = True

    # Debugging output
    #if state.debug:
    #    for i in removed_indices:
    #        print(f"Picked index {i} with probability {score_tensor[i].item()}")

    # Make a new state so we can modify it since it's mutable
   
    return create_copied_state(state, removed_array)


