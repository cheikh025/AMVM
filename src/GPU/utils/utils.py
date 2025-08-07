

import h5py
import numpy as np

import torch





def dequantize_squeezellm_3bit_matrix(qweights: torch.Tensor, original_weight_shape: tuple[int, int], lookup_table: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the 3-bit weight matrix stored by SqueezeLLM.
    Let m x n be the size of the original matrix (dequantized and untransposed)

    - qweights: should have shape (m // 32 * 3, n).T
    - lookup_table: should have shape (m, 8)

    In the SqueezeLLM code, the weights are stored in this format (for 3-bit):
    qweights[i, j] is a 32-bit integer
    qweights[i, j]:
bit:  012     345        678     ...   27 28 29               30 31
      ___     ___        ___     ...   ___                      __
    [i, j]  [i+1, j]  [i+2, j]        [i+9, j]      last 2 bits of [i+10, j]
    |_________________________________________|
                    30 bits

    qweights[i+1, j]:                                           31: lsb of [i+21, j]
                    _                    ___            ...  ...    _
    most significant bit of [i+10, j]   [i+11, j]

    qweights[i+2, j]                                            30 31 32
                    __                    ___          ___ ...    ___
    2 significant bits of [i+21, j]    [i+22, j]            [i+31, j]

    Cycle repeats for every 3 rows in qweights
    Note: qweights is transposed

    Returns the new tensor of size m
    """
    quantized_m, n = qweights.shape
    m = original_weight_shape[0]

    # Lookup table indices
    LUT_indices = np.zeros(original_weight_shape, dtype=np.uint32)
    LUT_indices = LUT_indices.T  # Transpose since it was transposed when being stored

    quantized_row = 0  # Row of quantized_m
    row = 0  # Row of actual m

    qweights = qweights.numpy().view(np.uint32)

    # Read in weights[row:row+32] from qweights[quantized_row]
    while quantized_row < quantized_m:
        # Read in weights[i:i+10]
        # All shifts should be in the opposite direction of quant.py in SqueezeLLM code
        for curr_row in range(row, row + 10):
            # Bitwise and 7, to get only the first 3 bits
            LUT_indices[curr_row] = (qweights[quantized_row] >> (3 * (curr_row - row))) & 7

        row += 10
        # Read in weights[i+10]
        # Read in 2 least significant bits (lsb) of row (row + 10)
        LUT_indices[row] |= (qweights[quantized_row] >> 30)
        quantized_row += 1

        # Then, the lsb of the next row of qweights is the msb of weights[i+10]
        LUT_indices[row] |= (qweights[quantized_row] & 1) << 2
        row += 1

        # Read in weights[i+11:i+21]
        for curr_row in range(row, row + 10):
            LUT_indices[curr_row] = (qweights[quantized_row] >> (3 * (curr_row - row) + 1)) & 7

        # Read in weights[i+21]
        row += 10
        # Read the lsb of weights[i+21]
        LUT_indices[row] |= (qweights[quantized_row] >> 31)
        quantized_row += 1

        # Read in the 2 msb's of weights[i+21] from the next quantized_row
        LUT_indices[row] |= (qweights[quantized_row] & 3) << 1
        row += 1

        # Read in weights[i+22:i+32]
        for curr_row in range(row, row + 10):
            LUT_indices[curr_row] = (qweights[quantized_row] >> (3 * (curr_row - row) + 2)) & 7

        quantized_row += 1
        row += 10

    # print(f"row is now {row}, quantized_row is now {quantized_row}")

    LUT_indices = LUT_indices.T
    LUT_indices = torch.from_numpy(LUT_indices.astype(np.int32))


    unquantized_weights = torch.zeros(original_weight_shape, dtype=torch.float16)
    for curr_row in range(0, m):
        unquantized_weights[curr_row, :] = lookup_table[curr_row, :][LUT_indices[curr_row, :]]

    # print(f"LUT[519, 405] is {LUT_indices[519, 405]}")
    return unquantized_weights


def get_dequantized_squeezellm_model(model, state_dict: dict[str, torch.Tensor]):
    # Replace this with layers to replace if not using OPT
    linear_layers = ('k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2')


    for name, module in model.named_modules():
        if name.endswith(linear_layers):
            qweight_name = name + '.qweight'
            LUT_name = name + '.lookup_table'
            unquantized_matrix = dequantize_squeezellm_3bit_matrix(state_dict[qweight_name], module.weight.shape, state_dict[LUT_name])
            module.weight.data = unquantized_matrix

            print(f"quantized module {name}")
    return model


def get_squeezellm_lookup_tables(model, state_dict: dict[str, torch.Tensor], sort_LUT: bool=True) -> dict[str, torch.Tensor]:
    """
    state_dict: the state dictionary saved from SqueezeLLM.
    If sorted, then the LUT will be sorted row-wise (this is one of our algorithm assumptions)
    Returns a dict with module_name -> lookup_table generated from SqueezeLLM.
    """
    # Replace this with layers to replace if not using OPT
    linear_layers = ('k_proj', 'v_proj', 'q_proj', 'out_proj', 'fc1', 'fc2')
    LUT_dict = {}
    for name, module in model.named_modules():
        if name.endswith(linear_layers):
            LUT_name = name + '.lookup_table'
            LUT = state_dict[LUT_name]
            if sort_LUT:
                # This is required for our algorithm, which assumes quantized values are sorted
                LUT, indices = torch.sort(LUT, dim=1)

            LUT_dict[name] = LUT

    return LUT_dict


def round_to_nearest_pole_sim(w: torch.Tensor, poles: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Taken from the SqueezeLLM library:
    w: weight values, 1 row
    poles: tuple of values - LUT
    Poles is the quantization grid.

    Round the numbers in w to the nearest value in poles.
    Returns the index in poles that it has rounded to
    """
    stack = []
    w = w.cpu()
    poles = poles.cpu()
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(dim=0)
    aug = torch.zeros_like(w, dtype=torch.int32)
    for i, c in enumerate(poles):
        aug += (idx == i) * i
    return aug.to(device)


def load_tensors_from_hdf5(filename) -> dict[str, np.ndarray]:
    """
    Load a dictionary of PyTorch tensors from an HDF5 file.

    Args:
    - filename (str): The filename of the HDF5 file to load the data from.

    Returns:
    - data_dict (dict): A dictionary where the keys are strings and the values are PyTorch tensors.
    """
    data_dict = {}
    with h5py.File(filename, "r") as f:
        # Iterate over datasets in the HDF5 file
        for key in f.keys():
            # Load dataset into numpy array
            data = np.array(f[key])
            # Add tensor to data_dict
            data_dict[key] = data
    return data_dict


def save_tensors_to_hdf5(data_dict, filename, subsets=None):
    """
    Save a dictionary of PyTorch tensors to an HDF5 file.

    Args:
    - data_dict (dict): A dictionary where the keys are strings and the values are PyTorch tensors.
    - filename (str): The filename of the HDF5 file to save the data to.
    """
    # Create a new HDF5 file
    with h5py.File(filename, "a") as f:
        # Create datasets for each tensor in data_dict
        for key, value in data_dict.items():
            if subsets is not None and key not in subsets:
                continue
            f.create_dataset(key, data=value)


def ReadData():
    # Assuming cached_weights is a dictionary of PyTorch tensors
    weights=load_tensors_from_hdf5("./data/cached_weights-1.h5")
    inputs=load_tensors_from_hdf5("./data/cached_inputs-1.h5")

    print(f"Weights (W) dimension: {weights['0.self_attn.k_proj'].shape}")
    print(f"Input (X) dimension: {inputs['0.self_attn.k_proj'].shape}")

    return weights, inputs



def calculate_inf_norm(W: torch.tensor, W_hat: torch.tensor, X: torch.tensor):
    """
    Calculate the difference between W_hat X^T and W X^T.
    Arguments:
    W: numpy array, shape (M,)
        Original weight vector.
    W_hat: numpy array, shape (M,)
        Quantized weight vector.
    X: numpy array, shape (K, M)
        Input matrix.
    Returns:
    Max Difference between W_hat X^T and W X^T.
    """
    # Calculate W_hat X^T
    W_hat_XT = torch.matmul(W_hat, X.T)
    # Calculate W X^T
    W_XT = torch.matmul(W, X.T)
    # Calculate the difference
    difference = abs(W_hat_XT - W_XT)

    diff = difference.detach().max().cpu().item()
    del difference
    del W_hat_XT
    del W_XT
    return diff


def calculate_inf_norm_avg(W: torch.tensor, W_hat: torch.tensor, X: torch.tensor):
    """
    Calculate the difference between W_hat X^T and W X^T.
    Arguments:
    W: numpy array, shape (M,)
        Original weight vector.
    W_hat: numpy array, shape (M,)
        Quantized weight vector.
    X: numpy array, shape (K, M)
        Input matrix.
    Returns:
    Max Difference between W_hat X^T and W X^T.
    """
    # Calculate W_hat X^T
    W_hat_XT = torch.matmul(W_hat, X.T)
    # Calculate W X^T
    W_XT = torch.matmul(W, X.T)
    # Calculate the difference
    difference = abs(W_hat_XT - W_XT)

    ret = difference.max(dim=1)[0].mean().cpu().item()
    del difference
    del W_hat_XT
    del W_XT
    return ret

def calculate_inf_norm_B_k(B_k: torch.tensor, W_hat: torch.tensor, X: torch.tensor):
    """
    Calculate the difference between W_hat X^T and B_k.
    Arguments:
    B_k: numpy array, shape (K,)
        Original weight vector.
    W_hat: numpy array, shape (M,)
        Quantized weight vector.
    X: numpy array, shape (K, M)
        Input matrix.
    Returns:
    Max Difference between W_hat X^T and W X^T.
    """
    # Calculate W_hat X^T
    W_hat_XT = torch.matmul(W_hat, X.T)
    # Calculate the difference
    # TODO: return non-abs value here for L biggest in array D

    # For calculating the array D (of size L of the largest )
    non_abs_difference = W_hat_XT - B_k

    difference = abs(non_abs_difference)

    # numpy.ndarray is not copied when returning, so this shouldn't affect performance if we calculate L
    # when we return

    return difference.max().item(), non_abs_difference


def calculate_l2_norm(W, W_hat, X):
    """
    Calculate the L2 norm of the difference between W_hat @ X.T and W @ X.T.

    Arguments:
    W: numpy array, shape (N,)
        Original weight vector.
    W_hat: numpy array, shape (N,)
        Quantized weight vector.
    X: numpy array, shape (M, N)
        Input matrix.

    Returns:
    L2 norm of the difference.
    """
    # Calculate W_hat @ X.T
    W_hat_XT = torch.matmul(W_hat, X.T)
    # Calculate W @ X.T
    W_XT = torch.matmul(W, X.T)
    # Calculate the difference
    difference = W_hat_XT - W_XT
    # Calculate the L2 norm of the difference
    l2_norm = torch.norm(difference)
    return l2_norm.cpu().item()


def calculate_l2_norm_matrix(W, W_hat, X):
    """
    Calculate the L2 norm of the difference between W_hat @ X.T and W @ X.T.

    Arguments:
    W: numpy array, shape (N,)
        Original weight vector.
    W_hat: numpy array, shape (N,)
        Quantized weight vector.
    X: numpy array, shape (M, N)
        Input matrix.

    Returns:
    L2 norm of the difference.
    """
    # Calculate W_hat @ X.T
    W_hat_XT = torch.matmul(W_hat, X.T)
    # Calculate W @ X.T
    W_XT = torch.matmul(W, X.T)
    # Calculate the difference
    difference = W_hat_XT - W_XT
    # Calculate the L2 norm of the difference
    l2_norms = torch.norm(difference, dim=1)

    ret = l2_norms.detach().mean().cpu().item()
    del W_hat_XT
    del W_XT
    del difference
    del l2_norms
    return ret
