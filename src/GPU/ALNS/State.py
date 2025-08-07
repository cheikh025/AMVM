import time

import utils.utils as utils

import numpy as np
import pandas as pd
import copy
import torch

# Solution file

"""These values define the behaviour of State's objective() function"""
FULL = 0  # Full evaluation: calculate the objective on the full sample
PARTIAL_D_SINGLE_CHANGE = 1  # Only recalculate based off of self.change (a single change), but don't change self.objective_value
FULL_D_SWAP = 2  # Recalculate the objective on the full eval, based off of a swap (used in local_search_swap)

RECALCULATE_FLAG = -1  # This is for self.objective_value, determines whether or not self.objective() does a full
# recalculation

class State:
    """
    Stores the current state

    Has the following variables (needs to be passed into init function):
    :var inputs: untransposed input matrix X, shape (nSamples, M), the sampled input
    :var weights: the current state of the row weights W_Q, int array of shape (M, )
    :var original_weights: unquantized weight row (row we want to minimize distance to), float array of shape (M, )
    Note: original_weights isn't necessarily in Q
    :var B_k: W @ X^T, stored to save computation time
    :var num_partial: int, number of differences stored in array D used for greedy repair; precondition: num_partial < inputs
    :var nQuantization: number of bits in the quantization
    :var num_levels: 2 ** nQuantization

    Calculated after being initialized:
    (These two should not be changed after initial row)
    :var wMin: lowest value in Q (quantized set)
    :var step: step size between elements in Q
    :var quantization_levels: set Q

    :var removed_array: a boolean array of shape (M, ) indicating which indices are removed

    If squeezellm:
    :var steps: tensor of size (num_levels - 1), the step size of going from steps[i] -> steps[i + 1]
        during computations, we will use steps to change the value of state.step

    :var keep_outliers: flag to not quantize outliers
    :var outlier_range: a weight is an outlier if the weight's magnitude is >= outlier_range

    """

    L_set: torch.tensor
    recalculate_flag: bool

    changes: list
    objective_value: float
    iteration: int
    debug: bool
    num_partial: int
    eval_flag: bool

    use_squeezellm: bool
    squeezellm_LUT: torch.Tensor

    keep_outliers: bool
    outlier_range: float

    outlier_mask: torch.Tensor

    def __init__(self, inputs: torch.tensor, weights: torch.tensor, original_weights: torch.tensor,
                 B_k: torch.tensor, nQuantization: int, num_partial: int, debug=False, LS_op=None,
                 torch_device=torch.device('cpu'), use_gptq=False, use_squeezellm=False, squeezellm_LUT:torch.Tensor=None,
                 keep_outliers=False, outlier_range=0.0, use_fir=False):
        """

        If use_squeezellm is on, then squeezellm_LUT has to be sorted, and weights should be tensor of integers
        """
        self.torch_device = torch_device
        # Variables for debugging
        self.iteration = 0
        self.cnt = 0
        self.objective_value = RECALCULATE_FLAG
        self.debug = debug

        self.use_squeezellm = use_squeezellm

        self.keep_outliers = keep_outliers
        self.outlier_range = outlier_range

        self.inputs = inputs
        self.weights = weights
        self.original_weights = original_weights
        self.B_k = B_k
        self.nQuantization = nQuantization

        # Mask which weights are outliers
        self.outlier_mask = torch.zeros_like(weights, device=torch_device)
        if self.keep_outliers:
            self.outlier_mask = original_weights.abs() >= self.outlier_range

            indices = torch.nonzero(self.outlier_mask, as_tuple=True)[0]
            print(f"Have {indices.numel()} outliers")
            # if len(indices) > 0:
            #     print(f"Outliers on indices {indices}, with weights {self.original_weights[indices]}")


        # Variables for Greedy Repair algorithm  -----------------------------------------------------------------------
        self.num_partial = min(num_partial, len(B_k))  # The number of differences (of inputs to B_k) we store for partial evaluation
        # (The size of L_set)
        self.eval_flag = FULL  # Used for full evaluation

        # TODO: remove, change is deprecated since this is only for change weight local search/greedy repair
        self.change = (-1, -1)  # Stores exactly one change for the partial eval of greedy repair:
        # Stores (value, index), meaning that weights[self.change[1]] has changed by self.change[0]
        # (-1, -1) means that there is no change
        # Both are integers: value is -1 or 1 since we only move up or down 1 step at a time

        self.changes = []  # List containing tuples (index, delta), meaning that W[index] changed by delta

        self.L_set = []  # This is a Pandas DataFrame with 3 columns: (index, signedD_k, absD_k), storing these properties for partial evaluation
        # index is the row number of the input (k) that produced D_k

        # --------------------------------------------------------------------------------------------------------------
        self.LS_op = LS_op  # Local search operators
        # either 'S', 'W'
        # S = swap, W = weights, anything else is none

        self.found_time = time.time()  # Time it took to find the current best solution (a time.time() object)

        # Calculate quantization variables
        self.wMin = torch.min(original_weights)
        self.wMax = torch.max(original_weights)
        self.num_levels = 2 ** nQuantization
        self.maxq = self.num_levels - 1

        self.step = (self.wMax - self.wMin) / (self.num_levels - 1)
        self.steps = None

        self.quantization_levels = self.wMin + torch.arange(self.num_levels, device=torch_device) * self.step
        if use_fir:
            a = -2.0 ** (nQuantization-1)
            b = 2.0 ** (nQuantization-1) - 1
            self.step = ((b - a) / (self.num_levels - 1)/(2.0 ** (nQuantization-1)))
            k_values = torch.arange(a, b + 1,  dtype=torch.float64, device=torch_device)  
            self.quantization_levels = k_values / (2.0 ** (nQuantization-1))
        if use_gptq:
            zero = torch.round(-self.wMin / self.step)
            q = torch.arange(self.num_levels, device=torch_device)

            self.quantization_levels = self.step * (q - zero)  # New quantization levels according to gptq

            # Dequantize based off how GPTQ does it
            self.weights = torch.clamp(torch.round(self.weights / self.step) + zero, 0, self.num_levels - 1).int()

        if use_squeezellm:
            # If we are using squeezeLLM, then this is non-uniform
            self.quantization_levels = squeezellm_LUT.to(torch_device)

            # Also, the step sizes are different, we need to store an array of 7 differences
            # Each difference step[i] corresponds to the step size when a weights goes from index i -> i + 1
            self.steps = self.quantization_levels[1:] - self.quantization_levels[:-1]


        # Array that marks which elements should be repaired (used by the remove operators)
        self.removed_array = torch.zeros(len(self.weights), dtype=torch.bool)  # Set this array to all False

        # Calculate the initial objective for the first time
        self.objective_value, non_abs_difference = utils.calculate_inf_norm_B_k(self.B_k,
                                                                                self.get_quantized_weights(),
                                                                                self.inputs)

        # If recalculate_flag, then the next objective call will do a recalculation
        self.recalculate_flag = False
        self.eval_flag = FULL  # Get full evaluation next time

        self.signedD_ks = non_abs_difference
        self.absD_ks = abs(non_abs_difference)

        self.L2_norm = torch.sum(self.absD_ks * self.absD_ks)

        # Get indices of the L_set
        L_set_ind = torch.topk(self.absD_ks, self.num_partial)[1]
        self.L_set = (self.signedD_ks[L_set_ind], L_set_ind)  # tuple of (values, ind)
        self.global_best = self.objective()  # Keep global best here

        # Calculations to optimize local search --------------------------
        # Calculate and store maxX and minX to prevent recalculation
        # TODO: since inputs are constant, we could pass this into each instance to prevent this recalculation
        self.maxX = torch.max(self.inputs)
        self.minX = torch.min(self.inputs)
        # Also calculate the min/max for each row
        # Discard the indices
        self.maxX_k = torch.max(self.inputs, dim=1)[0]
        self.minX_k = torch.min(self.inputs, dim=1)[0]


    def objective(self) -> float:
        # Returns value of infnorm(W @ X^T)
        if self.eval_flag == FULL:
            if len(self.changes) == 0:
                # If there are no changes, just return
                self.recalculate_flag = False
                return self.objective_value

            change_arr = torch.zeros(len(self.inputs), dtype=torch.float, device=self.torch_device)  # Store total changes here
            prev_step = self.step
            for idx, delta in self.changes:
                # Note: if squeezeLLM is on, then delta cannot be more than 1.
                if self.use_squeezellm:
                    # Calculate the new step, this is based on delta
                    # Integer weight is already updated
                    if delta == 1:
                        # q[w[i+1]] - q[w[i]]
                        self.step = self.steps[self.weights[idx] - 1]
                    elif delta == -1:
                        self.step = self.steps[self.weights[idx]]

                # Apply changes sequentially
                change_arr += torch.flatten(self.step * delta * self.inputs[:, idx])  # Apply these changes

            self.step = prev_step  # Return to old step to prevent any changes from affecting other code

            if self.recalculate_flag:  # Update D_ks and L_set only if we have recalculate flag
                self.changes.clear()  # Remove change queue

                # Apply the changes to the D_ks
                self.signedD_ks += change_arr
                self.absD_ks = torch.abs(self.signedD_ks)

                # Update L_set
                L_set_ind = self.absD_ks.topk(self.num_partial)[1]
                self.L_set = (self.signedD_ks[L_set_ind], L_set_ind)

                # Update objective value
                self.objective_value = abs(self.L_set[0][0]).item()

                self.L2_norm = torch.sum(self.absD_ks * self.absD_ks)

                # Debugging output
                if self.debug:
                    print(f"Evaluated value of new state: {self.objective_value}")
                    print(f"Evaluated l2 norm of new state: {self.L2_norm}")

                # Testing correctness for L_inf, L2 norm
                # When running tests, comment this out. This is slow
                # ------------------------------------------------------------------
                # print("Checking correctness")
                # test_obj_value, non_abs_difference = utils.calculate_inf_norm_B_k(self.B_k,
                #                                                                         self.get_quantized_weights(),
                #                                                                         self.inputs)

                # if abs(test_obj_value - self.objective_value) > 1e-3:
                #     print("objective value is incorrect!")
                #     print(f"Obj value: {self.objective_value}, actual val: {test_obj_value}")
                # print("Checking l2 norm")
                # actual_l2_norm = utils.calculate_l2_norm(self.original_weights, self.get_quantized_weights(), self.inputs) ** 2
                # if abs(actual_l2_norm - self.L2_norm) > 1:
                #     print("l2 norm value is incorrect!")
                #     print(f"Obj value: {actual_l2_norm}, actual val: {self.L2_norm}")
                # --------------------------------------------------------------------------


                self.recalculate_flag = False  # Next iteration doesn't need to recalculate
                return self.objective_value
            else:
                # TODO: change this back to just 1 return
                return torch.max(torch.abs(self.signedD_ks + change_arr)).item()  # Calculate these changes but don't apply it
        if self.eval_flag == PARTIAL_D_SINGLE_CHANGE:
            change_arr = torch.zeros(len(self.inputs), dtype=torch.float, device=self.torch_device)  # Store total changes here
            delta, idx = self.change    # extract from the tuple
            change_arr += torch.flatten(self.step * delta * self.inputs[:, idx])  # Apply these changes
            updated_signedD_ks = self.signedD_ks + change_arr
          #  Updated_absD_ks = torch.abs(updated_signedD_ks)
          #  new_L2_norm = torch.sum(Updated_absD_ks * Updated_absD_ks)
          #  if new_L2_norm > self.L2_norm:
          #      return np.inf
          #  # Compute the objective value based on the updated signedD_ks
            return torch.max(torch.abs(updated_signedD_ks)).item()
        return self.objective_value


    def get_quantized_weights(self) -> torch.tensor:
        """
        :return: quantized weights row of shape (M, )
        """
        if self.keep_outliers:
            # Add back in the outliers if this is the case
            return self.quantization_levels[self.weights] * (~self.outlier_mask) + self.original_weights * self.outlier_mask
        else:
            return self.quantization_levels[self.weights]



    def get_integer_weights(self, quantized_weights: torch.tensor) -> torch.tensor:
        """Returns q for each weight
        Returns a new integer array"""

        int_weights = torch.zeros(len(quantized_weights)).int()
        for i in range(len(quantized_weights)):
            int_weights[i] = round((quantized_weights[i] - self.wMin) / self.step)
        return int_weights

