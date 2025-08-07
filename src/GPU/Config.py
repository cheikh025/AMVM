from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    """
    Args:
        DBname: name of the matrix/tensor being quantized
        inputs: original activations passed into matrix
        weights: original weight matrix
        num_gpu: number of gpus used
        rows_per_gpu: number of instances per gpu
        index_iterations: number of rows to quantize (for debugging, usually it is all, i.e. weights.shape[0])
        use_multiprocess: flag to use multiprocessing (for debugging purposes)

        seconds: seconds to run per row
        nQuantized: number of bits

        save_weights: save weights to a file
        stored_weights_path: path to where we store the weights

        For debugging:
            save_to_file: save information per row to a file
            output_filename: path to where we save information per row

        nQuantized: number of bits

        use_gptq: flag to use gptq_matrix as starting weights
        gptq_matrix: starting weights from GPTQ

        use_squeezellm: flag to use squeezellm as starting weights
        squeezellm_LUT: lookup table associated with this matrix

        keep_outliers: flag to store outlier weights as fp16
        outlier_range: magnitude for a weight to be considered an outlier

        debug: debug flag during ALNS
        debug_process: prints debug info for code other than ALNS (process info, etc.)
    """
    DBname: str
    inputs: np.ndarray
    weights: np.ndarray
    num_gpu: int
    rows_per_gpu: int
    index_iterations: int
    use_multiprocess: bool

    seconds: int
    nQuantized: int

    save_weights: bool
    stored_weights_path: str

    save_to_file: bool
    output_filename: str

    use_gptq: bool
    gptq_matrix: torch.Tensor

    use_squeezellm: bool
    squeezellm_LUT: torch.Tensor

    keep_outliers: bool
    outlier_range: float

    debug: bool
    debug_process: bool