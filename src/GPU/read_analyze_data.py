"""
Class for debugging and data analysis

Use this to compute L2/L inf norm to store in a CSV file
Check src/README.md for how to use this file

"""
import gc

from RoundToNearest import FindNearest

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

import csv

from RoundToNearest import FindNearest

import numpy as np


def main() -> None:
    weight_file_path = "./full_data/cached_weights-transposed.h5"
    input_file_paths = ["./aug21_from_squeezellm.pt"]

    quantized_weights_path = "./full_data_output/aug21_full.h5"

    # Read all weights, then part1 then 2 of the inputs
    all_weights = ut.load_tensors_from_hdf5(weight_file_path)
    print(f"Read in weights")

    all_quantized_weights = ut.load_tensors_from_hdf5(quantized_weights_path)

    gptq_weights_path = "./aug21_squeezellm_rtn.h5"
    # gptq_weights_path = "./full_data_output/jul29_usingpaths.h5"
    all_gptq_weights = ut.load_tensors_from_hdf5(gptq_weights_path)
    print(f"Read in gptq weights from path {gptq_weights_path}")

    CSV_FILEPATH = "./Results/full_data/aug21_squeezellm_cpu.csv"
    CSV_FILE = open(CSV_FILEPATH, mode="a+")
    # with open(CSV_FILEPATH, mode="w") as CSV_FILE:
    writer = csv.writer(CSV_FILE)

    # Flag to calculate data for both our quantized weights and gptq_weights_path
    CALC_BOTH = False

    if CALC_BOTH:
        # headers = ["Matrix Name", "Quantized Inf Norm", "Average Quantized Inf Norm", "Avg quantized L2 norm",
        #            "GPTQ Inf Norm", "Average GPTQ Inf Norm", "Avg GPTQ L2 norm"]
        headers = ["Matrix Name", "Quantized Inf Norm", "Average Quantized Inf Norm", "Avg quantized L2 norm",
                   "RTN Inf Norm", "Average RTN Inf Norm", "Avg RTN L2 norm"]
    else:
        headers = ["Matrix Name", "Quantized Inf Norm", "Average Quantized Inf Norm", "Avg quantized L2 norm"]
    writer.writerow(headers)

    # Change this to use CPU if there's OOM issues
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    file_cnt = 0
    for input_file_path in input_file_paths:
        # if file_cnt == 1:
        #     file_cnt += 1
        #     continue
        gc.collect()
        torch.cuda.empty_cache()
        if input_file_path.endswith(".h5"):
            all_inputs = ut.load_tensors_from_hdf5(input_file_path)
        else:
            all_inputs = torch.load(input_file_path)
        print(f"Read inputs from path {input_file_path}")
        print(type(all_inputs))

        keys = all_inputs.keys()
        flag = 0


        for DBname in keys:
            gc.collect()
            torch.cuda.empty_cache()
            if 'fc1' not in DBname:
                continue

            # if flag == 0:
            #     flag += 1
            #     continue
            # if '11.fc2' not in DBname:
            #     continue
            print(f"Solving DBname {DBname}")
            if input_file_path.endswith(".pt"):
                inputs = torch.cat(all_inputs[DBname], dim=0)
                if 'fc' not in DBname:
                    inputs = inputs.flatten(0, 1)
                inputs = inputs.float().to(device)
            else:
                inputs = torch.tensor(all_inputs[DBname]).float().to(device)

            weights = torch.tensor(all_weights[DBname].T).float().to(device)

            # print(inputs.max(), inputs.min())
            # for lim in [0.01, 0.05, 0.1, 0.5, 1]:
            #     print(f"{lim}: {float(torch.sum(inputs.abs() < lim)) / inputs.numel()}")
            # continue

            # gptq_weights = torch.tensor(all_gptq_weights[DBname]).float().to(device)
            # TODO: NEED TO ALSO STORE ITERATIONS
            # inputs = all_inputs[DBname]
            # weights = all_weights[DBname].T
            # INDEX_ITERATIONS = inputs.shape[0]  # Quantize all rows
            try:
                # quantized_weights = ut.load_tensors_from_hdf5(quantized_weights_path)[DBname]
                quantized_weights = torch.tensor(all_quantized_weights[DBname]).float().to(device)
                gptq_weights = torch.tensor(all_gptq_weights[DBname]).float().to(
                    device)

                rows = quantized_weights.shape[0]

                inf_norm_nearest = 0
                avg_norm_nearest = 0
                inf_norm_quantized = 0
                avg_norm_quantized = 0
                avg_l2_norm_quantized = 0
                avg_l2_norm_nearest = 0
                num_rows = quantized_weights.shape[0]

                # if 'fc2' in DBname:
                #     num_rows = 20
                # else:
                #     num_rows = 40
                print("Reached here")
                inf_norm_quantized = ut.calculate_inf_norm(weights, quantized_weights, inputs)
                # print(1)
                avg_norm_quantized = ut.calculate_inf_norm_avg(weights, quantized_weights, inputs)
                # print(2)

                avg_l2_norm_quantized = ut.calculate_l2_norm_matrix(weights, quantized_weights, inputs)
                # print(3)


                if CALC_BOTH:
                    inf_norm_nearest = ut.calculate_inf_norm(weights, gptq_weights, inputs)
                    avg_norm_nearest = ut.calculate_inf_norm_avg(weights, gptq_weights, inputs)
                    avg_l2_norm_nearest = ut.calculate_l2_norm_matrix(weights, gptq_weights, inputs)

                    row = [DBname, inf_norm_quantized, avg_norm_quantized, avg_l2_norm_quantized, inf_norm_nearest,
                           avg_norm_nearest, avg_l2_norm_nearest]


                else:
                    row = [DBname, inf_norm_quantized, avg_norm_quantized, avg_l2_norm_quantized]
                print(row)
                # print(f"Inf norm for quantized is {ut.calculate_inf_norm(weights, quantized_weights, inputs)}, L2 norm quantized is {ut.calculate_l2_norm_matrix(weights, quantized_weights, inputs)},"
                #       f"Avg norm quantized = {ut.calculate_inf_norm_avg(weights, quantized_weights, inputs)}")

                writer.writerow(row)

                print(f"Inf norm for nearest is {inf_norm_nearest}, Inf norm quantized is {inf_norm_quantized}")
                print(f"DBname {DBname} quantized")

                del inf_norm_quantized
                del avg_l2_norm_quantized
                del avg_norm_quantized
                del inf_norm_nearest
                del avg_norm_nearest
                del avg_l2_norm_nearest
                gc.collect()
                torch.cuda.empty_cache()

            except Exception:
                # raise
                print(f"Error when quantizing {DBname}")
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

