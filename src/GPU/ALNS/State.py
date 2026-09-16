import time

import utils.utils as utils

import numpy as np
import pandas as pd
import copy
import torch
import random

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
                 keep_outliers=False, outlier_range=0.0, use_fir=False, discrete_domain=None,
                 acceptance_policy="linf", seed=0, fixed_mask=None):
        """

        If use_squeezellm is on, then squeezellm_LUT has to be sorted, and weights should be tensor of integers
        """
        self.torch_device = torch_device
        # Keep every input on the working device. The device may be auto-selected
        # (cuda -> mps -> cpu), so callers are not required to pre-move tensors.
        inputs = inputs.to(torch_device)
        weights = weights.to(torch_device)
        original_weights = original_weights.to(torch_device)
        B_k = B_k.to(torch_device)
        # Variables for debugging
        self.iteration = 0
        self.cnt = 0
        self.objective_value = RECALCULATE_FLAG
        self.debug = debug

        self.use_squeezellm = use_squeezellm

        self.keep_outliers = keep_outliers
        self.outlier_range = outlier_range
        if acceptance_policy not in {"linf", "linf_l2_tiebreak", "linf_l2_nonincrease"}:
            raise ValueError(f"Unknown acceptance policy: {acceptance_policy}")
        self.acceptance_policy = acceptance_policy
        self.python_rng = random.Random(seed)
        self.torch_generator = torch.Generator(device=torch_device).manual_seed(seed)

        self.inputs = inputs
        self.weights = weights.clone()
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
        self.fixed_mask = self.outlier_mask.bool()
        if fixed_mask is not None:
            fixed_mask = torch.as_tensor(fixed_mask, device=torch_device, dtype=torch.bool)
            if fixed_mask.shape != self.weights.shape:
                raise ValueError("Fixed-variable mask must match the initial solution")
            self.fixed_mask |= fixed_mask


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
            # Build the FIR levels on the working device (no CPU detour). MPS has
            # no float64, so use the input dtype (float32) there; other devices
            # keep float64 precision. Levels are cast to inputs.dtype afterwards.
            fir_dtype = inputs.dtype if torch_device.type == "mps" else torch.float64
            k_values = torch.arange(a, b + 1, dtype=fir_dtype, device=torch_device)
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


        if discrete_domain is not None:
            if use_gptq or use_squeezellm or use_fir:
                raise ValueError("Explicit domains cannot be combined with legacy domain modes")
            self.quantization_levels = torch.as_tensor(discrete_domain, device=torch_device,
                                                       dtype=inputs.dtype).clone()
        self.quantization_levels = self.quantization_levels.to(device=torch_device, dtype=inputs.dtype)
        if (self.quantization_levels.ndim != 1 or self.quantization_levels.numel() == 0
                or not torch.isfinite(self.quantization_levels).all()
                or (self.quantization_levels[1:] < self.quantization_levels[:-1]).any()):
            raise ValueError("The discrete domain must be a nonempty, finite, sorted vector")
        self.num_levels = self.quantization_levels.numel()
        self.maxq = self.num_levels - 1
        if self.weights.is_floating_point() and not torch.equal(self.weights, self.weights.round()):
            raise ValueError("Initial weights must be integer domain indices")
        self.weights = self.weights.long()
        if ((self.weights < 0) | (self.weights >= self.num_levels)).any():
            raise ValueError("Initial index outside discrete domain")
        self.steps = self.quantization_levels[1:] - self.quantization_levels[:-1]

        # Array that marks which elements should be repaired (used by the remove operators)
        self.removed_array = torch.zeros(len(self.weights), dtype=torch.bool, device=torch_device)  # Set this array to all False

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


    def move_residual(self, indices, new_indices):
        """Evaluate index assignments without mutating weights or residual caches."""
        indices = torch.as_tensor(indices, device=self.torch_device, dtype=torch.long).reshape(-1)
        new_indices = torch.as_tensor(new_indices, device=self.torch_device, dtype=torch.long).reshape(-1)
        if indices.shape != new_indices.shape or indices.unique().numel() != indices.numel():
            raise ValueError("A move must assign each variable at most once")
        if ((new_indices < 0) | (new_indices >= self.num_levels)).any():
            raise ValueError("Move outside discrete domain")
        if (self.fixed_mask[indices] & (new_indices != self.weights[indices])).any():
            raise ValueError("Move attempts to modify a fixed variable")
        deltas = self.quantization_levels[new_indices] - self.quantization_levels[self.weights[indices]]
        return self.signedD_ks + self.inputs[:, indices] @ deltas

    def evaluate_move(self, indices, new_indices):
        """Return the candidate infinity norm without applying the move."""
        return self.move_residual(indices, new_indices).abs().max().item()

    def _set_residual(self, residual):
        """Refresh all objective caches from a residual in the input dtype."""
        self.signedD_ks = residual
        self.absD_ks = residual.abs()
        rows = self.absD_ks.topk(self.num_partial).indices
        self.L_set = (residual[rows], rows)
        self.objective_value = self.absD_ks.max().item()
        self.L2_norm = residual.square().sum()

    def apply_move(self, indices, new_indices):
        """Commit one move and immediately refresh residual and objective caches."""
        residual = self.move_residual(indices, new_indices)
        self.weights[indices] = torch.as_tensor(new_indices, device=self.torch_device,
                                                dtype=self.weights.dtype)
        self._set_residual(residual)
        self.eval_flag, self.recalculate_flag = FULL, False
        return self.objective_value

    def accepts(self, candidate_linf, candidate_l2, *, incumbent_linf=None,
                incumbent_l2=None, epsilon=0.0):
        """Apply the configured infinity/L2 acceptance policy."""
        incumbent_linf = self.objective_value if incumbent_linf is None else incumbent_linf
        incumbent_l2 = self.L2_norm if incumbent_l2 is None else incumbent_l2
        linf_improves = candidate_linf < incumbent_linf - epsilon
        linf_ties = abs(candidate_linf - incumbent_linf) <= epsilon
        if self.acceptance_policy == "linf":
            return linf_improves
        if self.acceptance_policy == "linf_l2_tiebreak":
            return linf_improves or (linf_ties and candidate_l2 < incumbent_l2)
        return linf_improves and candidate_l2 <= incumbent_l2

    def objective(self) -> float:
        """Read the objective; support legacy pending-index changes for compatibility.

        New operators use evaluate_move/apply_move. Legacy queued edits are decoded
        directly so repeated and nonuniform edits cannot use an inferred step.
        """
        if self.eval_flag == PARTIAL_D_SINGLE_CHANGE:
            delta, idx = self.change
            return self.evaluate_move([idx], [self.weights[idx] + delta])
        if self.changes:
            residual = self.inputs @ self.get_quantized_weights() - self.B_k
            if not self.recalculate_flag:
                return residual.abs().max().item()
            self._set_residual(residual)
            self.changes.clear()
        self.recalculate_flag = False
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

        return (quantized_weights[:, None] - self.quantization_levels[None, :]).abs().argmin(dim=1)
