"""A solution class, returned from our custom ALNS call
"""

import numpy as np
import torch

import utils.utils as ut
from ALNS.State import State

import alns

import csv

class Solution:
    quantized_weights: torch.tensor
    inf_norm_value: float
    method_name: str
    nQuantized: int
    alns_object: alns.ALNS
    alns_result: alns.Result

    input_index: int

    best_state: State  # Only if we are returning from a regular ALNS solve call
    findNearestObj: float

    def __init__(self, quantized_weights: torch.tensor, inf_norm_value: float, method_name: str, nQuantized: int,
                 alns_object: alns.ALNS=None, result=None, solveTime=0, bestFoundSolutionTime=0):
        self.quantized_weights = quantized_weights
        self.inf_norm_value = inf_norm_value
        self.method_name = method_name
        self.nQuantized = nQuantized
        self.alns_object = alns_object
        self.alns_result = result
        self.input_index = 0   # Initialize row index
        self.initial_obj = 0
        self.solveTime = solveTime
        self.bestFoundSolutionTime = bestFoundSolutionTime


        self.row_index = 0


    def set_input_index(self, idx: int):
        self.input_index = idx

    def set_initial_obj(self, obj: float):
        self.initial_obj = obj

    def set_row_index(self, row_index: int):
        self.row_index = row_index

    def save_to_file(self, name, mode, print_headers=True, index_name="Input"):
        """Save this result to a file, with the name in the format:
        Results/output_<method_name>_<nQuantized>.csv


        """
        filename = f"./Results/full_data/output_{name}_{self.nQuantized}.csv"
        with open(filename, mode) as file:
            writer = csv.writer(file)  # CSV writer

            if print_headers:
                # Print headers
                header = [f"{index_name} Index", "Method Name", "Objective", "Initial Objective", "nQuantized"]

                if self.alns_object is not None:
                    # Add the settings and parameters if we have a state
                    header.append("Remove Operators")
                    header.append("Repair Operators")
                    header.append("LS Operator")
                    header.append("Num Iterations")
                    header.append("Best State Found Time")
                    header.append("Runtime")


                writer.writerow(header)
            results = [self.input_index, self.method_name, self.inf_norm_value, self.initial_obj, self.nQuantized]

            if self.alns_object is not None:
                remove_operators_tuples = self.alns_object.destroy_operators  # List of (name, operator) tuples
                remove_operators_list = [remove_operators_tuples[i][0] for i in range(len(remove_operators_tuples))]

                remove_operators_str = ';'.join(remove_operators_list)  # Destroy operators separated by ;

                # Do the same for repair operators
                repair_operators_tuples = self.alns_object.repair_operators
                repair_operators_list = [repair_operators_tuples[i][0] for i in range(len(repair_operators_tuples))]

                repair_operators_str = ';'.join(repair_operators_list)

                results.append(remove_operators_str)
                results.append(repair_operators_str)

                # Add LS Operators
                LS_op = "None"
                if self.alns_result.best_state.LS_op == 'S':
                    LS_op = "Swap"
                if self.alns_result.best_state.LS_op == 'W':
                    LS_op = "Change Weight"

                results.append(LS_op)

                results.append(len(self.alns_result.statistics.runtimes)) # Number of iterations
                results.append(self.bestFoundSolutionTime)  # Add time it took to find best state

                # Add runtime
                results.append(self.solveTime)

            # Write row to csv file
            writer.writerow(results)

    # def save_weights_hdf5(self, filename):
    #     """Save weights to src/result/filename.hdf5
    #     Appends to file
    #     """
    #     # Create a dictionary of {index: weight}
    #     dictionary = {self.row_index: self.quantized_weights}
    #     updated_filename = f"src/Results/quantized_values_{filename}_{self.input_index}_{self.nQuantized}.csv"
    #     ut.save_tensors_to_hdf5(dictionary, updated_filename)
