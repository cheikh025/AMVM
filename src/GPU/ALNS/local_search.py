"""Contains local search operators
These are run after running repair operators
"""
import time

import numpy as np

from ALNS.operator_utils import ALPHA_COEFFICIENT
from ALNS.State import *
import math
import copy
import torch
import gc
import random
import multiprocessing as mp

import csv
import sys

SWAP_EVAL_BATCHES = 3 # Number of batches
BIAS = 0.001  # Bias to account for precision issues in swap filtering
NUM_FILTERS = 100
EPSILON = 0.001  # Dealing with infinite loop issues

FILENAME = "onlyswap11.csv"

ij_diff_time = 0
check_cmax_min_time = 0
evaluate_swaps_time = 0
evaluate_swap_partial_time = 0
broadcasting_ijmask_time = 0
input_filter_time = 0
apply_swap_time = 0
new_t_calc = 0
diff_ij_calc = 0


total_linf_swap = 0
total_l2_swap = 0

swaps = {}

def debug():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(f"{type(obj)}, {obj.size()} on device {obj.device}")
        except Exception:
            pass

class LocalSearch:
    
    def __init__(self, state : State) -> None:
        self.state = state
        self.sorting_method = "Shuffle" # None , WorseScore
        self.use_filtering = True
        self.use_swap_filtering = True
        self.use_all_delta_q = False
        self.min_diff = self.state.minX_k - self.state.maxX_k
        self.max_diff = self.state.maxX_k - self.state.minX_k
        # Maximum difference between these two, perform abs to ensure positive
        # max_diff = abs(self.state.maxX - self.state.minX)  
        self.delta_q = 1
        self.t = self.state.objective()  
        self.D_kold = self.state.signedD_ks
        self.Cmin = []
        self.Cmax = []
        self.useful_input_indices = torch.arange(len(self.state.inputs), device=state.torch_device)
        self.sol_dict = {} # dictonary solution representation
        self.convert2_sol_dict()
        global NUM_FILTERS
        NUM_FILTERS = state.num_partial

        global swaps

    def run(self) -> None:
        """
        Performs local search on the current self.state. 
        Local search to be run after each repair operator
        """
        try:
            if self.state.LS_op == 'S':
                self.local_search_optimized_swap()
            if self.state.LS_op == 'W':
                self.local_search_optimized_swap()
                self.local_search_change_weights()
        except RuntimeError:
            print("Caught runtime error for run itself")

        return self.state


    # TODO: can remove, this is not used
    def local_search_change_weights(self) -> None:
        """Local search by trying to change a weight up or down by 1 step"""
        current_t = self.state.objective()

        while True:
            found_change = False
            indices = list(range(len(self.state.weights)))
            random.shuffle(indices)
          #  ## Calculate scores: Score(j) = max |a_kj| over active k
            #scores, _ = self.state.inputs[self.state.L_set[1]].abs().max(dim=0)
            ## Sort the scores in descending order to get the ranked indices
            #ranked_indices = torch.argsort(scores, descending=True)
            #idxs = torch.randperm(20)
            #indices = ranked_indices[:20]#[idxs]
            for i in indices:
                # print(f"Local search on iteration {i}")
                delta_i = 0
                new_partial_obj = math.inf
                if self.state.weights[i] > 0:
                    self.state.change = (-1, i)
                    self.state.eval_flag = PARTIAL_D_SINGLE_CHANGE
                    new_partial_obj = self.state.objective()
                    delta_i = -1

                if self.state.weights[i] + 1 < self.state.num_levels:
                    self.state.change = (1, i)
                    self.state.eval_flag = PARTIAL_D_SINGLE_CHANGE

                    inc_objective = self.state.objective()
                    if inc_objective < new_partial_obj:
                        delta_i = 1
                        new_partial_obj = inc_objective

                if new_partial_obj < current_t:

                    self.state.weights[i] += delta_i
                    self.state.changes.append((i, delta_i))

                    self.state.eval_flag = FULL
                    self.state.recalculate_flag = True  
                    new_objective = self.state.objective()

                    if new_objective < current_t:
                        # Update self.state and objective if this new change is better
                        if self.state.debug:
                            print(f"Local search found a better self.state with objective {new_partial_obj}")
                        found_change = True
                        current_t = new_objective
                        break
                    else:
                        # Otherwise, revert the changes
                        self.state.objective_value = current_t
                        self.state.weights[i] -= delta_i
                        self.state.changes.append((i, -delta_i))

                        # Do a full reevaluation on the new weights
                        self.state.recalculate_flag = True
                        # self.state.L_set = L_set_copy
                        self.state.eval_flag = FULL  # remember to change this to full to prevent self.state.objective() from doing a partial evaluation
                        self.state.objective()  

            if not found_change:
                break

    def convert2_sol_dict(self):
        """
        Converts weights into a dictionary, mapping value -> indices where weights[idx] == value
        """
        # First, make a dictionary of the indices, storing the indices of weights that have this quantized value
        # Initialize the dictionary
        for i in range(self.state.num_levels):
            self.sol_dict[i] = []  # Initialize empty list

        for index in range(len(self.state.weights)):
            # Only add the index if it's not an outlier.
            if not self.state.outlier_mask[index]:
                self.sol_dict[self.state.weights[index].item()].append(index)



    def input_filtering(self):
        if self.use_filtering:
            # self.D_kold = self.state.full_eval['signedD_k'].to_numpy()  # Reset D_kold
            self.D_kold = self.state.signedD_ks
            delta_q_step = self.delta_q * self.state.step
            Cmin = (-self.t - self.D_kold) / (delta_q_step)
            Cmax = (self.t - self.D_kold) / (delta_q_step)

            useful_conditions = (Cmin >= self.min_diff) | (Cmax <= self.max_diff)

            # Get the indices of the k's that don't have an impact on the filter checking
            self.useful_input_indices = useful_conditions.nonzero().flatten()

            if self.state.debug:
                print(f"Filtered out {self.state.inputs.shape[0] - self.useful_input_indices.numel()} out of {self.state.inputs.shape[0]} indices")


            self.Cmin = Cmin[self.useful_input_indices]
            self.Cmax = Cmax[self.useful_input_indices]
            self.D_kold = self.D_kold[self.useful_input_indices]


        else:
            delta_q_step = self.delta_q * self.state.step
            self.Cmin = (-self.t - self.D_kold) / (delta_q_step)
            self.Cmax = (self.t - self.D_kold) / (delta_q_step)


    def w1_sorting(self, unSortedW1s: list): 
        
        if self.sorting_method == "Shuffle":
            # Before doing evaluation for w1s, shuffle the unsorted ones first
            np.random.shuffle(unSortedW1s)
            return unSortedW1s
        
        elif self.sorting_method == "WorseScore":
            # TODO: This is not working yet, it should only sort a given w1 array
            # Make weights W into a row vector so that we can do matrix multiplication on it
            filtered_inputs = self.state.inputs[self.useful_input_indices] 
            row_weights = self.state.weights.reshape((1, -1))  # [ w_1  w_2  ...  w_m  ]
            
            # Compute the element-wise product of the broadcasted weight matrix and the partial input matrix
            element_wise_product = row_weights * filtered_inputs 
            # Our matrix now looks like this, with row weight vector broadcasted into k rows
            # [ w_1 * x_11  w_2 * x_12  ...  w_m * x_1m ]
            # [ ...                                ...  ]
            # [ w_1 * x_k1  w_2 * x_k2  ...  w_m * x_km ]

            self.D_kold = self.D_kold.reshape((-1, 1))  # Reshape into a k x 1 column vector
            
            # For each index jk, LB_jk = (w_jk - D_k - t) / x_kj
            # and UB_jk = (w_jk - D_k + t) / x_kj
            zeros = np.zeros_like(element_wise_product)  # Filter out the 0s from filtered_inputs

            # Do this filtering so if filtered_inputs == 0, the quotient is replaced with 0
            LB_matrix = np.divide((element_wise_product - (self.t + self.D_kold)), filtered_inputs, out=zeros, where=filtered_inputs != 0)
            UB_matrix = np.divide((element_wise_product - (-self.t + self.D_kold)), filtered_inputs, out=zeros, where=filtered_inputs != 0)

            LB_matrix -= row_weights  # Subtract row_weights (element_wise by row, broadcasted to a k x m matrix)
            UB_matrix -= row_weights

            LB_matrix = abs(LB_matrix)  # These matrices should now have the distance between w_j and its respective bound
            UB_matrix = abs(UB_matrix)

            dist_matrix = np.minimum(LB_matrix,
                                    UB_matrix) * -ALPHA_COEFFICIENT  # Take the min distance and multiply it by -a

            # Apply exp to dist_matrix to calculate score values
            individual_scores = np.exp(
                dist_matrix).T  # Also transpose it to a m * k matrix so we can do matmul with a k * 1 D_k vector

            abs_D_ks = abs(self.D_kold)

            score_array = np.matmul(individual_scores, abs_D_ks) / np.sum(abs_D_ks)  # R^T @ D_ks should give the scores
            # We divide by sum_abs_D_ks in order to get the weighted sum

            score_array = score_array.flatten()  # Change this back to a 1d array

            score_dict = {}
            for i in range(self.state.num_levels):
                score_dict[i] = []
            
            # Only calculate up to self.state.num_levels - 1 since we never swap last layer
            for q1 in range(self.state.num_levels - 1):
                # For each index, calculate the score
                for i in self.sol_dict[q1]:
                    score_dict[q1].append(score_array[i])

                # Then sort sol_dict[q1] by the score array
                self.sol_dict[q1] = [x for _, x in sorted(zip(score_dict[q1], self.sol_dict[q1]), reverse=False)]
        
        else:
            return unSortedW1s
        

    def local_search_optimized_swap(self):
        """New swap technique (checking necessary condition for swapping)"""
        found_swap = False
        gc.collect()
        torch.cuda.empty_cache()

        while True:
            loop_end = 2
            if self.use_all_delta_q:
                loop_end = self.state.num_levels
            for self.delta_q in range(1, loop_end):
                try:
                    found_swap = self.perform_swap()
                    if found_swap:
                        break
                except RuntimeError:
                    print("Caught an out of memory error")
                    torch.cuda.empty_cache()
                    return  # Stop if out of memory occurred
            if not found_swap:
                break  # If we couldn't find an improving swap, then stop


    def perform_swap(self) -> bool:
        """Attempt to perform swap
        Returns true if a successful swap is made
        Returns false if no swap was found

        Note: If use_squeezellm is on, then this only works if q2 = q1 + self.delta_q
        Also, if squeezellm is on, then we need to redo input filtering after every iteration of (q1, q2)
        since self.step will be different each time
        """
        if not self.state.use_squeezellm:
            self.input_filtering()  # Filter inputs here, after self.delta_q was selected
        # Main swap loops
        for q1 in range(self.state.num_levels - self.delta_q):
            if len(self.sol_dict[q1]) == 0:
                continue

            self.sol_dict[q1] = self.w1_sorting(self.sol_dict[q1])

            q2 = q1 + self.delta_q
            if len(self.sol_dict[q2]) == 0:
                continue

            if self.state.use_squeezellm:
                self.state.step = self.state.steps[q1]  # New step size
                self.input_filtering()  # Filter inputs here since the step size changes based on the weight

            # x_js tensor
            filtered_X_js = self.state.inputs[torch.meshgrid(self.useful_input_indices, torch.tensor(self.sol_dict[q2], device=self.state.torch_device), indexing='ij')]
            batch_size = min(len(self.sol_dict[q1]), 2048) # math.ceil(len(self.sol_dict[q1]) / SWAP_EVAL_BATCHES)  #min(len(self.sol_dict[q1]), 128) #
            nb_batch = math.ceil(len(self.sol_dict[q1]) /batch_size )
            curr_D_k = self.state.L_set[0][:NUM_FILTERS]  # k_e D_k values
            idx_ks = self.state.L_set[1][:NUM_FILTERS]  # k_e indices

            epsilons = (self.t - abs(curr_D_k)) / (self.delta_q * self.state.step) + BIAS  # Size k_e

           # shuffled_batch_indices = np.arange(SWAP_EVAL_BATCHES)
            shuffled_batch_indices = np.arange(nb_batch)
            np.random.shuffle(shuffled_batch_indices)
            # Using batches
            # for batch in range(SWAP_EVAL_BATCHES):
            for batch in shuffled_batch_indices:
                q1_batch = self.sol_dict[q1][batch * batch_size:min((batch + 1) * batch_size, len(self.sol_dict[q1]))]
                if len(q1_batch) == 0:
                    break
                filtered_X_is = self.state.inputs[torch.meshgrid(self.useful_input_indices, torch.tensor(q1_batch, device=self.state.torch_device), indexing='ij')]

                if self.use_swap_filtering:
                    # Initialize array of 1s
                    w1s = self.state.inputs[torch.meshgrid(idx_ks, torch.tensor(q1_batch, device=self.state.torch_device), indexing='ij')]

                    # Size k_e * |q2|
                    w2s = self.state.inputs[torch.meshgrid(idx_ks, torch.tensor(self.sol_dict[q2], device=self.state.torch_device), indexing='ij')]

                    # Size k_e * |q1| * |q2|
                    ij_diff = w1s[:, :, None] - w2s[:, None, :]

                    # Find the indices where curr_D_k is < 0, > 0 and apply the epsilon bound checking correctly
                    curr_D_k_neg_indices = torch.nonzero(curr_D_k < 0, as_tuple=True)
                    curr_D_k_pos_indices = torch.nonzero(curr_D_k >= 0, as_tuple=True)


                    ij_mask = torch.all(ij_diff[curr_D_k_pos_indices] < epsilons[curr_D_k_pos_indices][:, None, None], dim=0)
                    ij_mask &= torch.all(ij_diff[curr_D_k_neg_indices] > -epsilons[curr_D_k_neg_indices][:, None, None], dim=0)


                    # Go through each of the unfiltered pairs and check if it will improve objective
                    unfiltered_pairs_indices = torch.nonzero(ij_mask, as_tuple=True)


                    # Free pytorch tensors
                    # del ij_mask
                    # torch.cuda.empty_cache()

                    cand_pairs_w1, cand_pairs_w2 = (torch.tensor(q1_batch, device=self.state.torch_device)[unfiltered_pairs_indices[0]],
                                                    torch.tensor(self.sol_dict[q2], device=self.state.torch_device)[
                                                        unfiltered_pairs_indices[1]])

                else:
                    # TODO: delete, this is unused
                    diff = filtered_X_is[:, :, None] - filtered_X_js[:, None, :]
                    within_bounds = (self.Cmin[:, None, None] < diff) & (diff < self.Cmax[:, None, None])
                    row_cols_condition_holds = torch.all(within_bounds, dim=0)
                    possible_pairs = torch.nonzero(row_cols_condition_holds, as_tuple=True)

                    if len(possible_pairs[0]) == 0:
                        continue  # No possible pairs founds
                    cand_pairs_w1, cand_pairs_w2 = (torch.tensor(q1_batch, device=self.state.torch_device)[possible_pairs[0]],
                                                    torch.tensor(self.sol_dict[q2], device=self.state.torch_device)[possible_pairs[1]])

                best_w1, best_w2 = -1, -1
                best_obj = math.inf

                # Prevent from using too much memory, stop if the tensor created will be too big
                # 500m is the limit
                if len(self.useful_input_indices) * len(cand_pairs_w1) > 500_000_000:
                    print(f"Prevented swap filter matrix of size {len(self.useful_input_indices) * len(cand_pairs_w1)}")
                    return False

                if len(cand_pairs_w1) == 0:
                    continue
                # Get the new t value evaluated on filtered_ks for every potential swap
                diff_ijs = self.state.inputs[torch.meshgrid(self.useful_input_indices, cand_pairs_w1, indexing='ij')] - self.state.inputs[torch.meshgrid(self.useful_input_indices, cand_pairs_w2, indexing='ij')]
                new_t_on_filtered_ks = torch.max(torch.abs(self.D_kold[:, None] + (self.state.step * self.delta_q) * diff_ijs), dim=0)[0]

                del diff_ijs
                torch.cuda.empty_cache()

                if len(new_t_on_filtered_ks) == 0:
                    continue

                if self.state.debug:
                    print(f"We have {len(cand_pairs_w1)} candidate pairs")


                # USING L2 NORM TO FILTER OUT POTENTIAL SWAPS:
                ind_better = torch.nonzero(new_t_on_filtered_ks < self.t - EPSILON, as_tuple=True)[0]
                new_t_ind_better = new_t_on_filtered_ks[ind_better]
                if len(ind_better) == 0:
                    continue

                index_is, index_js = cand_pairs_w1[ind_better], cand_pairs_w2[ind_better]
                if self.state.debug:
                    print(f"We have {len(index_is)} candidate pairs after l_inf filtering")

                # TODO: check if this is needed, tries to prevent memory errors by limiting swap checks
                if len(index_is) >= 100:
                    index_is, index_js = index_is[:100], index_js[:100]
                # x_i - x_j for each candidate pair
                # Note: this is all_k * |cand_pairs|!

                diff_ijs = self.state.step * self.delta_q * (self.state.inputs[:, index_is] - self.state.inputs[:, index_js])
                # Each entry should now be the new signedD_k
                all_new_D_ks = diff_ijs + self.state.signedD_ks[:, None]

                # L2NORM: comment this out to stop filtering for L2 norm
                # --------------------------------------------------------------
                # Calculate the new L2 norm after each swap (L2_norm = sum of squared D_ks)
                new_L2_norms = torch.sum(all_new_D_ks * all_new_D_ks, dim=0)  # Sum over all columns

                # Find indices of swaps that improve the L2 norm
                ind_better_l2_and_l_inf = torch.nonzero(new_L2_norms <= self.state.L2_norm)
                #ind_better_l2_and_l_inf = torch.nonzero((new_L2_norms <= self.state.L2_norm) | (new_L2_norms >= self.state.L2_norm) )    # Skip L2

                if len(ind_better_l2_and_l_inf) == 0:
                    continue
                # ---------------------------------------------------------------

                # This should be the index of the swap of best L_inf improvement that also improves/keeps L2 norm
                best_inf_norm_swap_idx = torch.argmin(new_t_ind_better[ind_better_l2_and_l_inf])
                # print(f"Expected new L2 norm is {new_L2_norms[ind_better_l2_and_l_inf[best_inf_norm_swap_idx]]}")


                # Get idx of swap with smallest objective
                best_w1, best_w2 = index_is[ind_better_l2_and_l_inf[best_inf_norm_swap_idx]], index_js[ind_better_l2_and_l_inf[best_inf_norm_swap_idx]]

                # If there's still no improving swap, then continue
                if best_w1 == -1:
                    continue

                best_obj = self.evaluate_swap_general([best_w1], [best_w2])
                # Make sure it's < t - EPSILON to prevent precision issues going into infinite loop
                if best_obj < self.t - EPSILON:
                    new_obj = self.apply_swap_state([best_w1], q1, [best_w2])
                    self.t = new_obj
                    return True

        return False  # We couldn't find a swap, so return false

    def evaluate_swap_general(self, w1_list: list, w2_list: list) -> float:
        """Evaluates the swap if all element (w1) in q1 changes to q2, and (w2) -> q1"""

        if self.state.debug and len(self.state.changes) > 0:
            print("Unexpected changes stored when doing swap evaluation")

        for idx in w1_list:
            self.state.weights[idx] += self.delta_q
            self.state.changes.append((idx, self.delta_q))

        for idx in w2_list:
            self.state.weights[idx] -= self.delta_q
            self.state.changes.append((idx, -self.delta_q))

        self.state.recalculate_flag = True
        self.state.eval_flag = FULL
        new_obj = self.state.objective()  # Evaluate on these changes but don't apply them

        # Undo the changes
        for idx in w1_list:
            self.state.weights[idx] -= self.delta_q
            self.state.changes.append((idx, -self.delta_q))

        for idx in w2_list:
            self.state.weights[idx] += self.delta_q
            self.state.changes.append((idx, self.delta_q))

        # Recalculate
        self.state.recalculate_flag = True
        self.state.eval_flag = FULL
        self.state.objective()

        # Clear the changes list
        # This is probably unnecessary
        self.state.changes.clear()

        return new_obj

    def apply_swap_state(self, w1_list: list, q1: int, w2_list: list) -> float:
        """Applies the change w1_list -> q2, w2_list -> q1, assuming delta_q = 1
        Additionally also updates the dictionary, as well as D_kold
        Returns the new objective
        """
        q2 = q1 + self.delta_q
        for w1 in w1_list:
            self.state.changes.append((w1, self.delta_q))
            self.state.weights[w1] += self.delta_q

            self.sol_dict[q1].remove(w1)
            self.sol_dict[q2].append(w1)

        for w2 in w2_list:
            self.state.changes.append((w2, -self.delta_q))
            self.state.weights[w2] -= self.delta_q

            self.sol_dict[q2].remove(w2)
            self.sol_dict[q1].append(w2)

        self.state.recalculate_flag = True
        self.state.eval_flag = FULL
        new_obj = self.state.objective()  # Recalculate the objective
        self.D_kold = self.state.signedD_ks  # Update D_kold
        return new_obj

