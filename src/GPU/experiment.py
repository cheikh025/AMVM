"""
Don't use this file, it hasn't been updated
Can be deleted
"""


import sys

from RoundToNearest import FindNearest

# import sys
# print(sys.path)
# sys.path.insert(0, '../src')


from utils import utils as ut

from ALNS.ALNS import ALNS

from alns.stop import *
import torch
import math
import time
# import pycuda.driver as driver

import torch.multiprocessing as mp
# from multiprocessing import current_process
# Operators

row = 0


def CalcPrintNorms(X, originalW, Wq, name):
    naiveInfNorm = ut.calculate_inf_norm(originalW, Wq, X)
    print(f"Inf norm value of {name} WQ solution of row {row} :{naiveInfNorm}")
    # naiveL2Norm = ut.calculate_l2_norm(originalW, Wq, X)
    # print(f"L2 norm value of {name} WQ solution of row {row} :{naiveL2Norm}")



def executeALNS(iteration: int, nQuantized: int, output_filename: str, save_to_file: bool, save_weights: bool,
                seconds: int, debug: bool, device: torch.device, inputs, weights, index_type, lock):

    # print(f"Worker {current_process().pid} solving iteration {iteration}")

    # inputs = np.frombuffer(ROInput, dtype=np.float16).reshape(global_input_shape[0])
    # print(torch.cuda.is_available())
    sampledInput = inputs
    original_row = weights[0, :]

    if index_type == "Input":
        sampledInput = inputs[(iteration * nSamples):((iteration + 1) * nSamples)]
    else:
        original_row = weights[iteration, :]

    print(f"On experimentIteration {iteration}")

    print(f"input is on device {inputs.device}")

    nearWq, nearQ = FindNearest(original_row, nQuantized, device)

    print("Calculating inf_norm for nearest")
    # naiveInfNorm = ut.calculate_inf_norm(original_row, nearWq, sampledInput)
    # timeStart = time.time()

    # nearQ = np.zeros(768, dtype=int)
    # ALNS
    print("Starting...")
    alns_obj = ALNS(nearQ, original_row, sampledInput, nQuantized, debug=debug)
    alns_obj.set_stopping_criteria(stopping_criteria=MaxRuntime(seconds))
    # alns_obj.set_stopping_criteria(stopping_criteria=MaxIterations(seconds))
    alns_obj.set_LS_operator('S')
    alns_obj.set_torch_device(device)

    # alns_obj.set_num_partials(200)

    solution = alns_obj.solve()
    solution.set_input_index(iteration)

    CalcPrintNorms(sampledInput, original_row, solution.quantized_weights, "ALNS-Hill Climbing")

    mode = "a+"
    print_headers = False
    if iteration == 0:
        # mode = "w+"  # Overwrite the file first
        print_headers = True
    # For debugging purposes, don't print
    # Use experiment.py if you want to print
    # solution.save_to_file("jul2_one_delta_q_n3", "w+", print_headers=True)  # Save it to a file, print header if it's first iteration
    print(solution.alns_result.statistics.total_runtime)

    if save_to_file:
        print(f"Saving to file on iteration {iteration}")
        lock.acquire()
        solution.save_to_file(output_filename, mode,
                              print_headers=print_headers,
                              index_name=index_type)  # Save it to a file, print header if it's first iteration
        lock.release()
    # Save weights to file too
    if save_weights:
        solution.save_weights_hdf5(output_filename)
    # print(solution.alns_result.statistics.total_runtime)

    CalcPrintNorms(sampledInput, original_row, nearWq, "nearest")

# Global variables used for GPU
WEIGHTS_ARRAY = []
INPUTS_ARRAY = []

if __name__ == '__main__':
    # torch.cuda.init()
    # driver.init()
    global weights
    global inputs
    org_weights, org_inputs = ut.ReadData()
    DBname = list(org_weights.keys())[0]

    # torch.cuda.set_per_process_memory_fraction(0.2)


    org_weights = org_weights[DBname].T
    org_inputs = org_inputs[DBname]

    # Experiment Variables
    # START_ITERATION =
    INDEX_ITERATIONS = 20  # < 26
    OUTPUT_FILENAME = "20inputsswap1_2_40iterations"
    SAVE_WEIGHTS = False
    SAVE_TO_FILE = True
    SECONDS = 10
    DEBUG = False
    # Row runs it on full input, "Input" takes sampled input
    index_type = "Row"  # Either "Row" or "Input"


    # if len(sys.argv) > 2

    NUM_GPU = 4  # Keeping at 1 works if there's no GPU (uses cpu instead)
    ROWS_PER_GPU = 1

    for i in range(NUM_GPU):
        if torch.cuda.is_available():
            torch_device = torch.device(f'cuda:{i}')
        else:
            torch_device = torch.device('cpu')
        WEIGHTS_ARRAY.append(torch.from_numpy(org_weights).double().to(torch_device))
        INPUTS_ARRAY.append(torch.from_numpy(org_inputs).double().to(torch_device))

    mp.set_start_method('spawn')  # Use fork for multiprocessing
    lock = mp.Lock()


    nQuantized = 2
    # nSamples = 10000
    # nSamples = 2 ** 18
    # print(f"Using a sample of n={nSamples}")

    for i in range(1):
        if i == 0:
            index_type = "Row"  # Either "Row" or "Input"
            OUTPUT_FILENAME = "jul11_row20_5batches_shuffling_4gpu_5_1row_250s"
            SECONDS = 250
            args = []
            # torch_device = torch.device('cuda')
            # for iteration in range(INDEX_ITERATIONS):
            #     args.append((iteration, nQuantized, OUTPUT_FILENAME, SAVE_TO_FILE, SAVE_WEIGHTS, SECONDS, DEBUG,
            #                  torch_device, INPUTS_ARRAY[0], WEIGHTS_ARRAY[0], index_type))


        timeStart = time.time()
        # USING STARMAP --------------------------------------------------------
        # with mp.Pool(processes=NUM_CORES) as pool:
        #     pool.starmap(executeALNS, args)

        # WITHOUT MULTIPROCESSING --------------------------------------
        # for iteration in range(INDEX_ITERATIONS):
        #     executeALNS(*args[iteration])


        # EXPLICITLY CREATING PROCESSES ------------------------------------------
        LOOP_ITERATIONS = math.ceil(INDEX_ITERATIONS / ROWS_PER_GPU)

        for loop_iteration in range(LOOP_ITERATIONS):
            iteration_begin = ROWS_PER_GPU * NUM_GPU * loop_iteration
            iteration_end = min(ROWS_PER_GPU * NUM_GPU * (loop_iteration + 1), INDEX_ITERATIONS)

            child_processes = []

            for iteration in range(iteration_begin, iteration_end):
                GPU_IDX = iteration % NUM_GPU
                if torch.cuda.is_available():
                    torch_device = torch.device(f'cuda:{GPU_IDX}')
                    # torch_device = torch.device('cuda')
                else:
                    torch_device = torch.device('cpu')

                args = (iteration, nQuantized, OUTPUT_FILENAME, SAVE_TO_FILE, SAVE_WEIGHTS, SECONDS, DEBUG,
                        torch_device, INPUTS_ARRAY[GPU_IDX], WEIGHTS_ARRAY[GPU_IDX], index_type, lock)

                # Create all the child processes and start it

                child_process = mp.Process(target=executeALNS, args=args)
                child_process.start()
                child_processes.append(child_process)

                # print(f"reached here, iteration {iteration}")

            # Wait for all the child processes to finish running
            for child_process in child_processes:
                child_process.join()
                if DEBUG:
                    print(f"Child process {child_process} done")

        # Then continue
        print(f"Took {time.time() - timeStart} sec in total.")
