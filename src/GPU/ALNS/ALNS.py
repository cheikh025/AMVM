import alns as package_alns
import numpy as np
from alns.accept import *
from alns.select import *
from alns.stop import *

# Import operators
from .remove_operators import *
from .repair_operators import *
import torch

from typing import Optional, Any
# Solution file
from Solution import Solution
# Seed passed into ALNS algorithm, used for reproducibility
SEED = 9101

UNINITIALIZED_FLAG = -1  # flag for uninitialized values


# L_SET_PERCENTAGE = 0.005  # Percentage of the inputs given to L_set

# On accept/reject/best/better functions
class onReject:
    def __init__(self):
        self.__name__ = "onReject"
    def __call__(self, state: State, rnd_state):
        if state.debug:
            print(f"Rejected state with objective {state.objective()}")

class onBest:
    def __init__(self):
        self.__name__ = "onBest"
    def __call__(self, state: State, rnd_state):
        state.found_time = time.time()  # Store the time in the best state
        # local_search_change_weights(state)  # Run local search on state when we have a new global best
        if state.debug:
            state.global_best = state.objective()
            print(f"Current global best state found with objective {state.objective()}")



class ALNS:
    """ALNS wrapper class
    allows for certain settings
    """
    initial_row: torch.tensor
    original_row: torch.tensor
    inputs: torch.tensor
    B_k: torch.tensor  # Value of B_k; if not set, then it will calculate it when calling solve
    nQuantization: int
    debug: bool

    num_partials: int  # Size of L_set

    use_gptq: bool
    use_squeezellm: bool
    squeezellm_LUT: Optional[torch.Tensor]

    keep_outliers: bool
    outlier_range: float


    def __init__(self, initial_row: torch.tensor, original_row: torch.tensor, inputs: torch.tensor,
                nQuantization: int, use_gptq=False, use_squeezellm=False, squeezellm_LUT=None,
                keep_outliers=False, outlier_range=0.0, debug=False, use_fir=False):
        """

        :param initial_row: Initial row of size (M, ), integers (found from FindNearest)
        :param original_row: Original row from weight matrix, shape (M, )
        :param inputs: An (nSamples, m) size nparray containing input matrix X
        :param nQuantization: number of bits in the quantization
        :param debug=False: print debug output
        """
        self.initial_row = initial_row
        self.original_row = original_row
        self.inputs = inputs
        self.nQuantization = nQuantization
        self.debug = debug
        self.use_gptq = use_gptq
        self.use_squeezellm = use_squeezellm
        self.squeezellm_LUT = squeezellm_LUT
        self.use_fir = use_fir


        self.keep_outliers = keep_outliers
        self.outlier_range = outlier_range

        np.random.seed(SEED)
        self.rnd_state = np.random.RandomState(SEED)  # Random number generator passed into ALNS
        # self.rnd_state = np.random.RandomState()
        self.alns = package_alns.ALNS(self.rnd_state)  # Initialize ALNS class
        self.LS_op = None

        # Initial values: optional parameters
        # self.num_partials = int(len(inputs) * L_SET_PERCENTAGE)
        self.num_partials = 100
        self.B_k = UNINITIALIZED_FLAG
        self.stopping_criteria = MaxRuntime(250)  # Initial value: stopping criteria is 250 seconds
        self.acceptance_criteria = HillClimbing()  # Default value: hill climbing
        self.torch_device = torch.device('cpu')  # Either cpu or gpu




    def set_num_partials(self, num_partials: int):
        self.num_partials = num_partials

    def set_B_k(self, B_k: torch.tensor):
        self.B_k = B_k

    def set_stopping_criteria(self, stopping_criteria):
        """stopping_criteria is from alns.stop package"""
        self.stopping_criteria = stopping_criteria

    def set_acceptance_criteria(self, acceptance_criteria):
        """acceptance_criteria is from alns.accept package"""
        self.acceptance_criteria = acceptance_criteria


    def set_LS_operator(self, LS_op: str):
        """Set the local search operator
        'S': swap
        'W': change weight
        anything else means no LS operator
        """
        self.LS_op = LS_op

    def set_torch_device(self, torch_device: torch.device):
        """Either cpu or gpu"""
        self.torch_device = torch_device



    def solve(self) -> Solution:
        # Adding destroy/repair operators
        self.alns.add_destroy_operator(random_remove)
        self.alns.add_destroy_operator(worst_remove)
        self.alns.add_repair_operator(random_repair)
        self.alns.add_repair_operator(greedy_repair)

        # Select operator
        select = RouletteWheel(scores=[5, 2, 1, 0.5], decay=0.7, num_destroy=len(self.alns.destroy_operators),
                               num_repair=len(self.alns.repair_operators))


        # Also keep track of time
        timeStart = time.time()

        # Make initial state based on initial_row
        # Calculate B_k first to save computation time, if it hasn't been initialized yet
        if isinstance(self.B_k, int) and self.B_k == UNINITIALIZED_FLAG:
            self.B_k = torch.matmul(self.original_row, self.inputs.T)  # Do the matrix mult

        # Set the initial state
        initial_state = State(self.inputs, self.initial_row, self.original_row, self.B_k, self.nQuantization,
                              num_partial=self.num_partials, debug=self.debug, LS_op=self.LS_op,
                              torch_device=torch.device(self.torch_device), use_gptq=self.use_gptq,
                              use_squeezellm=self.use_squeezellm, squeezellm_LUT=self.squeezellm_LUT,
                              keep_outliers=self.keep_outliers, outlier_range=self.outlier_range, use_fir=self.use_fir)


        initial_objective = initial_state.objective()
        #self.acceptance_criteria = SimulatedAnnealing.autofit(init_obj=initial_objective,  worse=0.05, accept_prob=0.5, num_iters=200, method="exponential")

        # Debugging output: printing a message if we reject/accept a state
        reject_print = onReject()
        best_print = onBest()
        self.alns.on_reject(reject_print)
        self.alns.on_best(best_print)

        result = self.alns.iterate(initial_state, select, self.acceptance_criteria, self.stopping_criteria)  # Run ALNS algorithm
        best_state = result.best_state


        # Print out amount of time it took and time it took to get the best result
        solveTime = time.time() - timeStart
        bestFoundSolutionTime = result.best_state.found_time - timeStart


        # Make a solution class, populate it with the result and return it
        quantized_weights = best_state.get_quantized_weights()
        # inf_norm_value = utils.calculate_inf_norm_B_k(best_state.B_k, quantized_weights, best_state.inputs)[0]  # Correct
        inf_norm_value = best_state.objective()

        if self.debug:
            print(f"The best solution was found in {bestFoundSolutionTime} sec.")
            print(f"Solving one row takes {solveTime} sec.")


        method_name = "ALNS-Hill Climbing"

        solution = Solution(quantized_weights, inf_norm_value, method_name, self.nQuantization, alns_object=self.alns,
                            result=result, solveTime=solveTime, bestFoundSolutionTime=bestFoundSolutionTime)
        solution.best_state = best_state  # Store the best state in solution if we need to use it

        # Also, store the objective value of the FindNearest objective
        solution.set_initial_obj(initial_objective)

        return solution
