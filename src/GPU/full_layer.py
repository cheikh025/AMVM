from dataclasses import dataclass

import sys

from RoundToNearest import FindNearest, FindNearestNumpy

# import sys
# print(sys.path)
# sys.path.insert(0, '../src')




from utils import utils as ut

from ALNS.ALNS import ALNS
import EGLS.EGLS as EGLS

import time
from alns.stop import *
import torch
import torch.multiprocessing as mp
import math
import os

from Config import Config

import gc

from typing import Iterable, List, Optional

import copy
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity


from Solution import Solution

from signal import signal, SIGINT


def CalcPrintNorms(X, originalW, Wq, name, iteration):
    naiveInfNorm = ut.calculate_inf_norm(originalW, Wq, X)
    print(f"Inf norm value of {name} WQ solution of row {iteration} :{naiveInfNorm}")
    # naiveL2Norm = ut.calculate_l2_norm(originalW, Wq, X)
    # print(f"L2 norm value of {name} WQ solution of row {row} :{naiveL2Norm}")


# TODO: change this to use config instead of taking args explicitly
def executeALNS(iteration: int, nQuantized: int, output_filename: str, save_to_file: bool, save_weights: bool,
                seconds: int, debug: bool, device: torch.device, inputs: torch.tensor, weights: torch.tensor,
                use_gptq=False, gptq_weights: torch.tensor=None, use_squeezellm=False, squeezellm_LUT: torch.Tensor=None,
                keep_outliers: bool=False, outlier_range: float=0.0):
    """
    weights is one row (m, )
    See Config.py class for what these variables mean
    """
    # Comment these lines out if you want to use CPU when there is GPU support
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if squeezellm_LUT is not None:
        squeezellm_LUT = squeezellm_LUT[iteration]
        print(f"Squeezellm on iteration {iteration} is {squeezellm_LUT}")

    solution = None
    try:
        sampledInput = inputs
        original_row = weights

        print(f"On experimentIteration {iteration}")

        if use_gptq:
            nearQ = gptq_weights
        elif use_squeezellm:
            nearQ = ut.round_to_nearest_pole_sim(weights, squeezellm_LUT, device)

        else:
            nearWq, nearQ = FindNearest(original_row, nQuantized, device)

        # ALNS
        print("Starting...")
        alns_obj = ALNS(nearQ, original_row, sampledInput, nQuantized, debug=debug, use_gptq=use_gptq, use_squeezellm=use_squeezellm, squeezellm_LUT=squeezellm_LUT,
                        keep_outliers=keep_outliers, outlier_range=outlier_range)
        # Set the number of seconds here, using swap, and the device
        alns_obj.set_stopping_criteria(stopping_criteria=MaxRuntime(seconds))

        alns_obj.set_LS_operator('S')
        alns_obj.set_torch_device(device)

        # Solve
        solution = alns_obj.solve()
        solution.set_input_index(iteration)


        print(f"Inf norm value of row {iteration} is {solution.inf_norm_value}")
        print(f"L2 norm of row {iteration} is {solution.alns_result.best_state.L2_norm}")


        mode = "a+"
        print_headers = False
        if iteration == 0:
            mode = "a+"  # Overwrite the file first
            print_headers = True
        # For debugging purposes, don't print
        # Use experiment.py if you want to print

        print(f"Solving iteration {iteration} took {solution.alns_result.statistics.total_runtime}s")
        # This is how many iterations this row has ran for
        print(f"Row {iteration} has these runtime length {len(solution.alns_result.statistics.runtimes)}")

        if save_to_file:
            print(f"Saving to file on iteration {iteration}")
            try:
                solution.save_to_file(output_filename, mode,
                                      print_headers=print_headers,
                                      index_name="Row")  # Save it to a file, print header if it's first iteration
            except Exception:
                print("Caught error in saving to file")

        # Save weights to file too
        if save_weights:
            # lock.acquire()
            data_dict = {f"{iteration}": solution.quantized_weights.cpu().numpy()}
            print(f"Saving iteration {iteration} to tmp file")
            weight_file_path = f"./full_data_output/tmp/tmp_matrix{iteration}.h5"
            # Delete file contents first
            try:
                os.remove(weight_file_path)
            except Exception as e:
                if debug:
                    print("error with removing weight file path")
            ut.save_tensors_to_hdf5(data_dict, weight_file_path)

    except Exception as e:
        # Error with running
        print(f"Solving row {iteration} raised an issue, not writing to file")
        print(e)
    finally:
        print(f"Freeing cache for iteration {iteration}")
        torch.cuda.empty_cache()


def quantize_indices_concurrently(indices: list[int] | np.ndarray, INPUTS_ARRAY: list[torch.Tensor], WEIGHTS_ARRAY: list[torch.Tensor], config: Config):
    """Takes in iterable indices and runs quantization algorithm on each row index specified"""
    TOTAL_ITERATIONS = len(indices)
    # EXPLICITLY CREATING PROCESSES ------------------------------------------
    LOOP_ITERATIONS = math.ceil(TOTAL_ITERATIONS / config.rows_per_gpu / config.num_gpu)

    for loop_iteration in range(LOOP_ITERATIONS):
        timeStart = time.time()
        iteration_begin = config.rows_per_gpu * config.num_gpu * loop_iteration
        iteration_end = min(config.rows_per_gpu * config.num_gpu * (loop_iteration + 1), TOTAL_ITERATIONS)

        child_processes = []

        for iteration in range(iteration_begin, iteration_end):
            GPU_IDX = iteration % config.num_gpu
            if torch.cuda.is_available():
                torch_device = torch.device(f'cuda:{GPU_IDX}')
            else:
                torch_device = torch.device('cpu')


            row_idx = indices[iteration]
            torch_weight_row = WEIGHTS_ARRAY[GPU_IDX][row_idx, :]
            gptq_row = config.gptq_matrix

            if config.use_gptq:
                # TODO: can put this in GPU beforehand
                gptq_row = config.gptq_matrix[row_idx, :].clone().to(torch_device)
            args = (row_idx, config.nQuantized, config.output_filename, config.save_to_file, config.save_weights, config.seconds, config.debug,
                    torch_device, INPUTS_ARRAY[GPU_IDX], torch_weight_row, config.use_gptq, gptq_row, config.use_squeezellm, config.squeezellm_LUT,
                    config.keep_outliers, config.outlier_range)

            # Create all the child processes and start it
            if config.use_multiprocess:
                child_process = mp.Process(target=executeALNS, args=args)
                child_processes.append(child_process)
            else:
                executeALNS(*args)

        for child_process in child_processes:
            child_process.start()
        # Wait for all the child processes to finish running
        for child_process in child_processes:
            child_process.join()
            if config.debug_process:
                print(f"Child process {child_process} done")
        print(
            f"This iteration of {iteration_end - iteration_begin}, iteration_begin is {iteration_begin}, iteration_end is {iteration_end} rows took {time.time() - timeStart} sec")

    print(f"Iteration ended")



def quantize_matrix(config: Config) -> np.ndarray:
    """
    Quantizes the matrix given in config.weights using config.inputs

    Saves the matrices as a .h5 file in the path config.

    Returns the quantized matrix as a numpy array.
    """

    WEIGHTS_ARRAY = []
    INPUTS_ARRAY = []

    for i in range(config.num_gpu):
        if torch.cuda.is_available():
            torch_device = torch.device(f'cuda:{i}')
        else:
            torch_device = torch.device('cpu')
        wt = torch.from_numpy(config.weights).float().detach().to(torch_device)
        inputs_arr = torch.from_numpy(config.inputs).detach().float().to(torch_device)
        WEIGHTS_ARRAY.append(wt)
        INPUTS_ARRAY.append(inputs_arr)

    # Start timer (for debugging)
    timeStart = time.time()

    unquantized_indices = np.arange(config.index_iterations)

    print(f"Using {config.rows_per_gpu} rows per gpu")
    quantize_indices_concurrently(unquantized_indices, INPUTS_ARRAY, WEIGHTS_ARRAY, config)

    # Read in every row from the temporary file, store in numpy array then save numpy array
    quantized_matrix = np.zeros(config.weights.shape)  # initialize matrix with same size

    incomplete_row_indices = []

    CNT_FIND_NEAREST = 0
    for row_idx in range(config.index_iterations):
        row_filename = f"./full_data_output/tmp/tmp_matrix{row_idx}.h5"

        try:
            dict = ut.load_tensors_from_hdf5(row_filename)
            for key, quantized_row in dict.items():
                quantized_matrix[row_idx, :] = quantized_row
        except Exception as e:
            # Row was not quantized, then add this to incomplete row_indices
            incomplete_row_indices.append(row_idx)

        # Remove temporary file
        try:
            os.remove(row_filename)
        except Exception as e:
            if config.debug_process:
                print("error with removing filename")

    # Attempt to run the algorithm again on the incomplete row indices
    # If there's any error, replace it with nearest solution

    config.rows_per_gpu = max(config.rows_per_gpu // 2, 1)  # Only do half of the rows per gpu to prevent memory errors
    # ROWS_PER_GPU = 2  # Only do 2 rows per gpu if it failed
    print(f"using {config.rows_per_gpu} rows per gpu")
    if incomplete_row_indices != []:
        print(f"Incomplete row indices is not empty, it has size {len(incomplete_row_indices)}")
        quantize_indices_concurrently(unquantized_indices, INPUTS_ARRAY, WEIGHTS_ARRAY, config)
    # TODO: put this in function instead of reusing code
    for incomplete_row_index in incomplete_row_indices:
        print(f"Retrying incomplete row index {incomplete_row_index}")
        row_filename = f"./full_data_output/tmp/tmp_matrix{incomplete_row_index}.h5"
        try:
            dict = ut.load_tensors_from_hdf5(row_filename)
            for key, quantized_row in dict.items():
                quantized_matrix[incomplete_row_index, :] = quantized_row
            print(f"Successfully quantized incomplete row idx {incomplete_row_index}")
        except Exception as e:
            # Row was not quantized, get nearest value and store solution
            if config.save_weights:
                nearWq, nearQ = FindNearestNumpy(config.weights[incomplete_row_index, :], config.nQuantized)
                quantized_matrix[incomplete_row_index] = nearWq
                print(f"Resorted to findnearest for row idx {incomplete_row_index}")
                CNT_FIND_NEAREST += 1

    if config.save_weights:
        # Then store the quantized matrix in the file
        data_dict = {config.DBname: quantized_matrix}
        ut.save_tensors_to_hdf5(data_dict, config.stored_weights_path)
        print(f"Stored DB {config.DBname}")

        print(f"Used {CNT_FIND_NEAREST} ways of find_nearest")

    # Then continue
    print(f"Quantizing db {config.DBname} took {time.time() - timeStart} sec in total.")
    del WEIGHTS_ARRAY
    del INPUTS_ARRAY

    return quantized_matrix



def get_config_settings() -> Config:
    """
    Returns the config settings without DBname, inputs, weights, index iterations or gptq matrix
    """
    USE_MULTIPROCESS = True
    SAVE_WEIGHTS = True  # Save weights to a .h5 file

    # This should be for debugging/analysis purposes only
    SAVE_TO_FILE = False  # Flag to store L_inf norm in a csv file

    SECONDS = 10
    DEBUG = False
    DEBUG_PROCESS = False
    # Row runs it on full input, "Input" takes sampled input

    # Note: at most one of these should be true
    use_gptq = False
    use_squeezellm = True

    NUM_GPU = 4  # Keeping at 1 works if there's no GPU (uses cpu instead)
    ROWS_PER_GPU = 10
    nQuantized = 3
    OUTPUT_FILENAME = f"aug20_test_fix"  # This is for displaying L_inf norm in the csv value
    # Note: This is stored in Results/full_data/output_{OUTPUT_FILENAME}


    USE_OUTLIERS = True
    OUTLIER_RANGE = 0.2496

    stored_weights_path = "./full_data_output/aug22_full.h5"

    return Config(None, None, None, NUM_GPU, ROWS_PER_GPU, 0, USE_MULTIPROCESS, SECONDS, nQuantized, SAVE_WEIGHTS,
                  stored_weights_path, SAVE_TO_FILE, OUTPUT_FILENAME, use_gptq, None, use_squeezellm, None,
                  USE_OUTLIERS, OUTLIER_RANGE, DEBUG, DEBUG_PROCESS)


def quantize_given_matrix(DBname: str, inputs: np.ndarray, weights: np.ndarray,
                          gptq_weights_path: str=None, squeezellm_LUT: torch.Tensor=None) -> np.ndarray:
    """
    Quantizes the given matrix given the data
    If use_gptq is true in the config settings, then read in the gptq weights from gptq_weights_path
    """
    torch.set_grad_enabled(False)
    mp.set_start_method('forkserver', force=True)  # Use fork for multiprocessing


    config = get_config_settings()
    config.DBname = DBname
    config.inputs = inputs
    config.weights = weights

    gptq_matrix = None
    if config.use_gptq and gptq_weights_path is not None:
        gptq_matrix = torch.from_numpy(ut.load_tensors_from_hdf5(gptq_weights_path)[DBname])
        print(f"Loaded in GPTQ matrix from filepath {gptq_weights_path}")

    config.gptq_matrix = gptq_matrix
    config.squeezellm_LUT = squeezellm_LUT

    index_iterations = weights.shape[0]
    config.index_iterations = index_iterations

    # Specify how many rows per gpu depending on matrix
    if 'fc2' in DBname:
        ROWS_PER_GPU = 5
    else:
        ROWS_PER_GPU = 10
    
    config.rows_per_gpu = ROWS_PER_GPU
    config.index_iterations = config.weights.shape[0]

    # # Use this for testing, only runs for 1 batch
    # print("Only solving for 1 batch")
    # config.index_iterations = config.rows_per_gpu * config.num_gpu

    print(f"Solving DBname {DBname}")
    args = (config,)
    if config.use_multiprocess:
        # Use multiprocess here to prevent memory leaks by stopping the main process after it's done
        process = mp.Process(target=quantize_matrix, args=args)
        process.start()
        process.join()
    else:
        quantize_matrix(*args)


    # Read in and return the new quantized matrix
    all_weights = ut.load_tensors_from_hdf5(config.stored_weights_path)
    return all_weights[DBname]


def get_testing_settings() -> Config:
    config = get_config_settings()
    config.use_multiprocess = False
    config.num_gpu = 1
    config.rows_per_gpu = 1
    config.seconds = 30
    config.debug = False
    config.debug_process = False

    config.stored_weights_path = "./testing_results/aug2_l2_testing_prelim.h5"
    OUTPUT_FILENAME = f"aug1_l2_norm_test_no_early"
    config.output_filename = OUTPUT_FILENAME
    config.save_to_file = False
    config.save_weights = False

    return config

# TODO: remove this, this was only for debugging purposes
def full_layer_path():
    torch.set_grad_enabled(False)
    mp.set_start_method('forkserver', force=True)  # Use fork for multiprocessing


    gptq_weights_file_path = "./gptq_asym_weights_trick.h5"  # TODO: fill this in
    args = []

    weight_file_path = "./full_data/cached_weights-transposed.h5"  # Already transposed
    input_file_paths = ["./full_data/cached_inputs_part1.h5", "./full_data/cached_inputs_part2.h5", ]

    # config = get_config_settings()
    config = get_testing_settings()


    # stored_input_paths = ["./full_data_output/quantized_inputs_part1.h5", "./full_data_output/quantized_inputs_part2.h5"]


    all_gptq_matrix = None
    if config.use_gptq:
        all_gptq_matrix = ut.load_tensors_from_hdf5(gptq_weights_file_path)
        print(f"Loaded in GPTQ matrix from filepath {gptq_weights_file_path}")

    # TODO: check if this is on
    # TODO: put this as a config setting
    REMOVE_WEIGHT_FILE = False  # Change this to true to remove the .h5 file
    if REMOVE_WEIGHT_FILE:
        try:
            os.remove(config.stored_weights_path)
        except Exception:
            if config.debug_process:
                print("error with removing filename")

    # Read all weights, then part1 then 2 of the inputs
    all_weights = ut.load_tensors_from_hdf5(weight_file_path)
    print(f"Read in weights")
    # print(len(all_weights.keys()))
    matrix_iter = 0

    for input_file_path in input_file_paths:

        all_inputs = ut.load_tensors_from_hdf5(input_file_path)
        print(f"Read inputs from path {input_file_path}")

        keys = all_inputs.keys()
        # keys = all_weights.keys()
        for DBname in keys:
            # if 'k_proj' not in DBname:
            #     continue

            print(f"Solving DBname {DBname}")

            if 'fc2' in DBname:
                ROWS_PER_GPU = 5
            else:
                ROWS_PER_GPU = 10

            if config.rows_per_gpu == 1:
                ROWS_PER_GPU = 1

            inputs = all_inputs[DBname]
            # inputs = np.zeros((768, 768))
            weights = all_weights[DBname].T
            # INDEX_ITERATIONS = weights.shape[0]  # Quantize all rows
            INDEX_ITERATIONS = 3 * ROWS_PER_GPU * config.num_gpu
            gptq_matrix = None
            if config.use_gptq:
                gptq_matrix = torch.from_numpy(all_gptq_matrix[DBname])

            # TODO: create a config class to store everything
            # TODO: use multiprocessing here in order to prevent memory leaks

            # config = Config(DBname, inputs, weights, NUM_GPU, ROWS_PER_GPU, INDEX_ITERATIONS,
            #                 USE_MULTIPROCESS, SECONDS, nQuantized, SAVE_WEIGHTS, stored_weights_path, SAVE_TO_FILE,
            #                 OUTPUT_FILENAME, use_gptq, gptq_matrix, DEBUG, DEBUG_PROCESS)
            config.DBname = DBname
            config.inputs = inputs
            config.weights = weights
            config.gptq_matrix = gptq_matrix
            config.index_iterations = INDEX_ITERATIONS
            config.rows_per_gpu = ROWS_PER_GPU

            args = (config,)
            if not config.use_multiprocess:
                quantize_matrix(*args)
            else:
                process = mp.Process(target=quantize_matrix, args=args)
                process.start()
                process.join()


if __name__ == '__main__':
    # Experiment Variables
    torch.set_grad_enabled(False)
    mp.set_start_method('forkserver')  # Use fork for multiprocessing
    full_layer_path()

