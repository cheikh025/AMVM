"""Contains local search operators
These are run after running repair operators
"""
import numpy as np

from ALNS import counters, tuning
from ALNS.operator_utils import ALPHA_COEFFICIENT
from ALNS.State import FULL, State
import torch

NUM_FILTERS = 100
EPSILON = 0.001  # Dealing with infinite loop issues

def run_local_search(state: State) -> State:
    """Run a local-search pass unless the starting point is already settled.

    A settled point is one an earlier pass examined without finding an improving
    move, and nothing has moved since. Whether a pass finds a move is a property
    of the point rather than of the random draws, because the pass evaluates
    every candidate and keeps the best, so a repeat pass on the same point is
    guaranteed to find nothing. Skipping it changes later random draws, so the
    skip is statistically equivalent rather than trajectory identical and is
    judged at equal time.
    """
    if tuning.SKIP_SETTLED_LOCAL_SEARCH and state.settled:
        counters.bump("local_search_skipped")
        return state
    counters.bump("local_search_ran")
    return LocalSearch(state).run()


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
        # One transfer of the domain. Level comparisons then happen on the host,
        # where they cost nothing, instead of synchronizing once per level pair.
        self.level_values = self.state.quantization_levels.tolist()
        self.sol_dict = {} # dictonary solution representation
        self.convert2_sol_dict()
        global NUM_FILTERS
        NUM_FILTERS = state.num_partial

    def run(self) -> None:
        """
        Performs local search on the current self.state. 
        Local search to be run after each repair operator
        """
        moves_before = self.state.move_count
        if self.state.LS_op == 'S':
            self.local_search_optimized_swap()
            # The swap loop only exits once a full pass finds no improving move,
            # so the point it leaves behind is verified.
            self.state.settled = True
        if self.state.LS_op == 'W':
            self.local_search_optimized_swap()
            self.local_search_change_weights()
            # Weight moves can reopen the swap neighborhood, so this point is
            # only verified for the combined operator when nothing moved at all.
            self.state.settled = self.state.move_count == moves_before
        counters.bump("local_search_passes")

        return self.state


    def local_search_change_weights(self) -> None:
        """Apply best adjacent single-variable moves under the selected policy."""
        while True:
            best = None
            mutable = torch.nonzero(~self.state.fixed_mask, as_tuple=True)[0]
            indices = mutable.tolist()
            levels = dict(zip(indices, self.state.weights[mutable].tolist()))
            self.state.python_rng.shuffle(indices)
            for i in indices:
                current = levels[i]
                for candidate in (current - 1, current + 1):
                    if not 0 <= candidate < self.state.num_levels:
                        continue
                    residual = self.state.move_residual([i], [candidate])
                    linf = residual.abs().max().item()
                    l2 = residual.square().sum()
                    if self.state.accepts(linf, l2, epsilon=EPSILON):
                        proposal = (linf, l2.item(), i, candidate)
                        if best is None or proposal[:2] < best[:2]:
                            best = proposal
            if best is None:
                break
            _, _, i, candidate = best
            self.state.apply_move([i], [candidate])

    def convert2_sol_dict(self):
        """
        Converts weights into a dictionary, mapping value -> indices where weights[idx] == value
        """
        # First, make a dictionary of the indices, storing the indices of weights that have this quantized value
        # Initialize the dictionary
        for i in range(self.state.num_levels):
            self.sol_dict[i] = []  # Initialize empty list

        # One transfer for every level instead of one reduction per level. The
        # per-level lists stay in increasing index order, as before.
        mutable = torch.nonzero(~self.state.fixed_mask, as_tuple=True)[0]
        for index, level in zip(mutable.tolist(), self.state.weights[mutable].tolist()):
            self.sol_dict[level].append(index)



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
            shuffled = list(unSortedW1s)
            self.state.python_rng.shuffle(shuffled)
            return shuffled
        
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

        while True:
            loop_end = 2
            if self.use_all_delta_q:
                loop_end = self.state.num_levels
            for self.delta_q in range(1, loop_end):
                found_swap = self.perform_swap()
                if found_swap:
                    break
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
        if tuning.DEVICE_RESIDENT_SWAP:
            return self._perform_swap_device_resident()
        return self._perform_swap_bounded()

    def _perform_swap_bounded(self, variable_tile=None, row_tile=None,
                              candidate_chunk=None) -> bool:
        """Find the best admissible swap using bounded two-dimensional tiles."""
        variable_tile = tuning.VARIABLE_TILE if variable_tile is None else variable_tile
        candidate_chunk = tuning.CANDIDATE_CHUNK if candidate_chunk is None else candidate_chunk
        if row_tile is None:
            row_tile = tuning.resolve_row_tile(len(self.state.inputs), candidate_chunk,
                                               self.state.inputs.element_size())
        best = None
        screen_rows = self.state.L_set[1][:NUM_FILTERS]

        for q1 in range(self.state.num_levels - self.delta_q):
            q2 = q1 + self.delta_q
            q1_values = self.w1_sorting(self.sol_dict[q1])
            q2_values = self.sol_dict[q2]
            if not q1_values or not q2_values:
                continue
            physical_delta = self.state.quantization_levels[q2] - self.state.quantization_levels[q1]
            if physical_delta == 0:
                continue

            for i_start in range(0, len(q1_values), variable_tile):
                index_i = torch.as_tensor(q1_values[i_start:i_start + variable_tile],
                                          device=self.state.torch_device, dtype=torch.long)
                for j_start in range(0, len(q2_values), variable_tile):
                    index_j = torch.as_tensor(q2_values[j_start:j_start + variable_tile],
                                              device=self.state.torch_device, dtype=torch.long)
                    survivor_mask = torch.ones((len(index_i), len(index_j)), dtype=torch.bool,
                                               device=self.state.torch_device)
                    if self.state.acceptance_policy == "linf_l2_tiebreak":
                        screen_bound = self.t + EPSILON
                    else:
                        screen_bound = self.t - EPSILON
                    for row_start in range(0, len(screen_rows), row_tile):
                        rows = screen_rows[row_start:row_start + row_tile]
                        pair_delta = (self.state.inputs[rows][:, index_i, None]
                                      - self.state.inputs[rows][:, None, index_j])
                        candidate = (self.state.signedD_ks[rows, None, None]
                                     + physical_delta * pair_delta)
                        survivor_mask &= torch.all(candidate.abs() <= screen_bound, dim=0)
                        if not survivor_mask.any():
                            break

                    survivor_i, survivor_j = torch.nonzero(survivor_mask, as_tuple=True)
                    for start in range(0, len(survivor_i), candidate_chunk):
                        ii = index_i[survivor_i[start:start + candidate_chunk]]
                        jj = index_j[survivor_j[start:start + candidate_chunk]]
                        maxima = torch.zeros(len(ii), dtype=self.state.inputs.dtype,
                                            device=self.state.torch_device)
                        squares = torch.zeros_like(maxima)
                        for row_start in range(0, len(self.state.inputs), row_tile):
                            rows = slice(row_start, row_start + row_tile)
                            residuals = (self.state.signedD_ks[rows, None]
                                         + physical_delta
                                         * (self.state.inputs[rows, ii] - self.state.inputs[rows, jj]))
                            maxima = torch.maximum(maxima, residuals.abs().max(dim=0).values)
                            squares += residuals.square().sum(dim=0)

                        if self.state.acceptance_policy == "linf_l2_tiebreak":
                            admissible = ((maxima < self.t - EPSILON)
                                          | ((maxima - self.t).abs() <= EPSILON)
                                          & (squares < self.state.L2_norm))
                        else:
                            admissible = maxima < self.t - EPSILON
                        if self.state.acceptance_policy == "linf_l2_nonincrease":
                            admissible &= squares <= self.state.L2_norm
                        if not admissible.any():
                            continue
                        candidates = torch.nonzero(admissible, as_tuple=True)[0]
                        tile_linf = maxima[candidates]
                        minimum = tile_linf.min()
                        tied = candidates[tile_linf == minimum]
                        chosen = tied[torch.argmin(squares[tied])]
                        proposal = (minimum.item(), squares[chosen].item(),
                                    int(ii[chosen]), int(jj[chosen]), q1, q2)
                        if best is None or proposal[:2] < best[:2]:
                            best = proposal

        if best is None:
            return False
        _, _, i, j, q1, q2 = best
        self.state.apply_move([i, j], [q2, q1])
        self.sol_dict[q1].remove(i)
        self.sol_dict[q2].append(i)
        self.sol_dict[q2].remove(j)
        self.sol_dict[q1].append(j)
        self.t = self.state.objective_value
        self.D_kold = self.state.signedD_ks
        return True

    def _perform_swap_device_resident(self, variable_tile=None, row_tile=None,
                                      candidate_chunk=None) -> bool:
        """Find the best admissible swap with one synchronization per pass.

        Selection is identical to :meth:`_perform_swap_bounded`: the best pair by
        infinity norm, ties broken by the sum of squares, and the earliest
        candidate kept when both agree. The difference is where the comparison
        happens. The bounded form copies four scalars back to the host for every
        candidate chunk and asks the host which is better; here the running best
        is six device tensors updated by a masked comparison, so the host learns
        the answer once, after the whole pass.
        """
        state = self.state
        device, dtype = state.torch_device, state.inputs.dtype
        variable_tile = tuning.VARIABLE_TILE if variable_tile is None else variable_tile
        candidate_chunk = tuning.CANDIDATE_CHUNK if candidate_chunk is None else candidate_chunk
        if row_tile is None:
            row_tile = tuning.resolve_row_tile(len(state.inputs), candidate_chunk,
                                               state.inputs.element_size())
        screen_rows = state.L_set[1][:NUM_FILTERS]
        infinity = torch.tensor(float("inf"), device=device, dtype=dtype)

        best_linf, best_l2 = infinity.clone(), infinity.clone()
        best_pair = torch.full((4,), -1, device=device, dtype=torch.long)

        if state.acceptance_policy == "linf_l2_tiebreak":
            screen_bound = self.t + EPSILON
        else:
            screen_bound = self.t - EPSILON
        # Candidates above this can never be accepted, whatever the incumbent is.
        acceptance_bound = torch.tensor(
            self.t + EPSILON if state.acceptance_policy == "linf_l2_tiebreak"
            else self.t - EPSILON, device=device, dtype=dtype)

        for q1 in range(state.num_levels - self.delta_q):
            q2 = q1 + self.delta_q
            q1_values = self.w1_sorting(self.sol_dict[q1])
            q2_values = self.sol_dict[q2]
            if not q1_values or not q2_values:
                continue
            if self.level_values[q2] == self.level_values[q1]:
                continue  # host comparison: an empty move cannot improve anything
            physical_delta = state.quantization_levels[q2] - state.quantization_levels[q1]
            levels = torch.tensor([q1, q2], device=device, dtype=torch.long)

            for i_start in range(0, len(q1_values), variable_tile):
                index_i = torch.as_tensor(q1_values[i_start:i_start + variable_tile],
                                          device=device, dtype=torch.long)
                for j_start in range(0, len(q2_values), variable_tile):
                    index_j = torch.as_tensor(q2_values[j_start:j_start + variable_tile],
                                              device=device, dtype=torch.long)
                    survivor_mask = torch.ones((len(index_i), len(index_j)), dtype=torch.bool,
                                               device=device)
                    for row_start in range(0, len(screen_rows), row_tile):
                        rows = screen_rows[row_start:row_start + row_tile]
                        pair_delta = (state.inputs[rows][:, index_i, None]
                                      - state.inputs[rows][:, None, index_j])
                        candidate = (state.signedD_ks[rows, None, None]
                                     + physical_delta * pair_delta)
                        survivor_mask &= torch.all(candidate.abs() <= screen_bound, dim=0)

                    # One synchronization per variable tile, to size the survivor list.
                    survivor_i, survivor_j = torch.nonzero(survivor_mask, as_tuple=True)
                    counters.bump("swap_screened", survivor_mask.numel())
                    counters.bump("swap_survivors", len(survivor_i))
                    for start in range(0, len(survivor_i), candidate_chunk):
                        ii = index_i[survivor_i[start:start + candidate_chunk]]
                        jj = index_j[survivor_j[start:start + candidate_chunk]]
                        if tuning.PRUNE_TO_INCUMBENT:
                            # Nothing above the incumbent can win, and nothing above
                            # the acceptance bound can be taken at all.
                            maxima, squares = self._exact_candidate_scores_pruned(
                                ii, jj, physical_delta,
                                torch.minimum(best_linf, acceptance_bound))
                        else:
                            maxima, squares = self._exact_candidate_scores(
                                ii, jj, physical_delta, row_tile)
                        counters.bump("swap_exact_evaluations", len(ii))

                        admissible = self._admissible(maxima, squares)
                        linf_key = torch.where(admissible, maxima, infinity)
                        l2_key = torch.where(admissible, squares, infinity)
                        # Best in this chunk: lowest infinity norm, ties by sum of
                        # squares, earliest candidate when both agree.
                        chunk_linf = linf_key.min()
                        tied = torch.where(linf_key == chunk_linf, l2_key, infinity)
                        position = tied.argmin()
                        chunk_l2 = tied[position]

                        better = ((chunk_linf < best_linf)
                                  | ((chunk_linf == best_linf) & (chunk_l2 < best_l2)))
                        best_linf = torch.where(better, chunk_linf, best_linf)
                        best_l2 = torch.where(better, chunk_l2, best_l2)
                        chosen = torch.cat([ii[position, None], jj[position, None], levels])
                        best_pair = torch.where(better, chosen, best_pair)

        i, j, q1, q2 = best_pair.tolist()  # the pass synchronizes here, once
        if i < 0:
            return False
        state.apply_move([i, j], [q2, q1])
        self.sol_dict[q1].remove(i)
        self.sol_dict[q2].append(i)
        self.sol_dict[q2].remove(j)
        self.sol_dict[q1].append(j)
        self.t = state.objective_value
        self.D_kold = state.signedD_ks
        counters.bump("swaps_applied")
        return True

    def _exact_candidate_scores(self, ii, jj, physical_delta, row_tile):
        """Return the exact infinity norm and sum of squares of each candidate."""
        state = self.state
        maxima = torch.zeros(len(ii), dtype=state.inputs.dtype, device=state.torch_device)
        squares = torch.zeros_like(maxima)
        for row_start in range(0, len(state.inputs), row_tile):
            rows = slice(row_start, row_start + row_tile)
            residuals = (state.signedD_ks[rows, None]
                         + physical_delta
                         * (state.inputs[rows, ii] - state.inputs[rows, jj]))
            maxima = torch.maximum(maxima, residuals.abs().max(dim=0).values)
            squares += residuals.square().sum(dim=0)
        return maxima, squares

    def _exact_candidate_scores_pruned(self, ii, jj, physical_delta, prune_bound):
        """Exact scores, abandoning candidates that can no longer win.

        The pass keeps the candidate with the smallest infinity norm, and a running
        maximum only ever grows, so once a candidate's partial maximum passes the
        best one found so far it cannot be chosen and its remaining rows are dead
        work. Candidates are dropped between row stages, and the stage grows as the
        live set shrinks, which holds the intermediate tensor near a fixed size
        rather than letting it collapse into many tiny launches.

        The comparison is strict, so a candidate that can still tie the incumbent
        survives and remains available to win on the sum of squares. Dropped
        candidates keep a partial maximum, which is a lower bound already past the
        bound, so the caller still rejects them; their sums of squares are partial
        and are never read.
        """
        state = self.state
        device = state.torch_device
        total_rows = len(state.inputs)
        live = torch.arange(len(ii), device=device)
        maxima = torch.zeros(len(ii), dtype=state.inputs.dtype, device=device)
        squares = torch.zeros_like(maxima)

        budget_elements = max(1, tuning.SWAP_MEMORY_BUDGET_MB * 1024 * 1024
                              // (state.inputs.element_size() * tuning.LIVE_INTERMEDIATES))
        start, stage = 0, min(tuning.PRUNE_FIRST_STAGE, total_rows)
        while start < total_rows and len(live):
            rows = slice(start, start + stage)
            columns_i, columns_j = ii[live], jj[live]
            residuals = (state.signedD_ks[rows][:, None]
                         + physical_delta * (state.inputs[rows][:, columns_i]
                                             - state.inputs[rows][:, columns_j]))
            maxima[live] = torch.maximum(maxima[live], residuals.abs().max(dim=0).values)
            squares[live] += residuals.square().sum(dim=0)
            counters.bump("pruned_row_candidate_products",
                          (min(start + stage, total_rows) - start) * len(live))

            start += stage
            if start < total_rows:
                # One synchronization per stage buys dropping every candidate that
                # can no longer win.
                live = live[maxima[live] <= tuning.prune_slack(prune_bound)]
                if len(live):
                    stage = int(min(max(tuning.PRUNE_FIRST_STAGE,
                                        min(stage * tuning.PRUNE_STAGE_GROWTH,
                                            budget_elements // len(live))),
                                    total_rows - start))
        counters.bump("prune_stages", 1)
        return maxima, squares

    def _admissible(self, maxima, squares):
        """Apply the configured acceptance policy to a vector of candidates."""
        if self.state.acceptance_policy == "linf_l2_tiebreak":
            admissible = ((maxima < self.t - EPSILON)
                          | ((maxima - self.t).abs() <= EPSILON)
                          & (squares < self.state.L2_norm))
        else:
            admissible = maxima < self.t - EPSILON
        if self.state.acceptance_policy == "linf_l2_nonincrease":
            admissible = admissible & (squares <= self.state.L2_norm)
        return admissible

    def evaluate_swap_general(self, w1_list: list, w2_list: list) -> float:
        """Evaluates the swap if all element (w1) in q1 changes to q2, and (w2) -> q1"""
        indices = list(w1_list) + list(w2_list)
        new_indices = ([int(self.state.weights[i]) + self.delta_q for i in w1_list]
                       + [int(self.state.weights[i]) - self.delta_q for i in w2_list])
        return self.state.evaluate_move(indices, new_indices)

    def apply_swap_state(self, w1_list: list, q1: int, w2_list: list) -> float:
        """Applies the change w1_list -> q2, w2_list -> q1, assuming delta_q = 1
        Additionally also updates the dictionary, as well as D_kold
        Returns the new objective
        """
        q2 = q1 + self.delta_q
        for w1 in w1_list:
            self.sol_dict[q1].remove(w1)
            self.sol_dict[q2].append(w1)

        for w2 in w2_list:
            self.sol_dict[q2].remove(w2)
            self.sol_dict[q1].append(w2)

        indices = list(w1_list) + list(w2_list)
        new_indices = [q2] * len(w1_list) + [q1] * len(w2_list)
        new_obj = self.state.apply_move(indices, new_indices)
        self.D_kold = self.state.signedD_ks  # Update D_kold
        return new_obj
