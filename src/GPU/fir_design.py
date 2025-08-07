from ALNS.ALNS import ALNS
from RoundToNearest import FindNearest
from alns.stop import *
import torch
import time
import copy
torch.set_printoptions(precision=8)
import numpy as np
from scipy.signal import remez, lfilter, freqz
from fir_config import project_specs



def design_ideal_remez_filter(order: int, ftype: str, edges, fs: float = 2.0, K: float = 1.0, gd: int = 16) -> np.ndarray:
    """
    Designs an optimal continuous-coefficient FIR filter using the Remez algorithm.
    This serves as the ideal baseline before quantization.
    
    Returns:
        The full impulse response (h_full) of the ideal filter.
    """
    # Process bands and desired for each filter type
    if ftype.lower() == 'lowpass':
        pass_edge, stop_edge = edges
        bands = [0, pass_edge, stop_edge, fs/2]
        desired = [K, 0]
    elif ftype.lower() == 'highpass':
        stop_edge, pass_edge = edges
        bands = [0, stop_edge, pass_edge, fs/2]
        desired = [0, K]
    elif ftype.lower() == 'bandpass':
        stop1, pass1, pass2, stop2 = edges
        bands = [0, stop1, pass1, pass2, stop2, fs/2]
        desired = [0, K, 0]
    elif ftype.lower() in ['notch', 'bandstop']:
        pass1, stop1, stop2, pass2 = edges
        bands = [0, pass1, stop1, stop2, pass2, fs/2]
        desired = [K, 0, K]
    else:
        raise ValueError(f"Unknown filter type '{ftype}'.")

    h_full = remez(numtaps=order + 1, bands=bands, desired=desired, grid_density=gd, fs=fs)
    return h_full

def get_remez_edges_from_specs(specs):
    ftype = specs['filter_type'].lower()
    pe = specs.get('pass_edge', None)
    se = specs.get('stop_edge', None)
    pe2 = specs.get('pass_edge2', None)
    se2 = specs.get('stop_edge2', None)
    fs = specs.get('fs', 2.0)
    # Decide edges for each type:
    if ftype == 'lowpass':
        edges = (pe, se)
    elif ftype == 'highpass':
        edges = (se, pe)
    elif ftype == 'bandpass':
        # Both pass_edge2 and stop_edge2 must not be None
        if pe2 is None or se2 is None:
            raise ValueError("For bandpass, specify both 'pass_edge2' and 'stop_edge2'")
        edges = (se, pe, pe2, se2)
    elif ftype in ('notch', 'bandstop'):
        if pe2 is None or se2 is None:
            raise ValueError("For bandstop/notch, specify both 'pass_edge2' and 'stop_edge2'")
        edges = (pe, se, se2, pe2)
    else:
        raise ValueError(f"Unknown filter type '{ftype}'.")
    return edges



def make_fir_desired(
    num_points,
    ftype,
    edges,
    fs=2.0,
    K=1.0,
):
    """
    Generate angular frequency grid and desired response vector for FIR filter design.

    Parameters
    ----------
    num_points : int
        Number of grid points.
    ftype : str
        Filter type: 'lowpass', 'highpass', 'bandpass', 'notch'/'bandstop'.
    edges : tuple or list
        Frequency edges (normalized to fs=2.0):
        - For 'lowpass':      (pass_edge, stop_edge)
        - For 'highpass':     (stop_edge, pass_edge)
        - For 'bandpass':     (stop1, pass1, pass2, stop2)
        - For 'notch':        (pass1, stop1, stop2, pass2)
        These match the convention in Scipy's remez and classic DSP books.
    fs : float
        Sampling frequency for normalization. Default 2.0.
    K : float
        Passband gain (default 1.0).

    Returns
    -------
    angular_omega_grid : ndarray
        Grid of frequencies in radians (0 to pi).
    d_desired : ndarray
        Desired frequency response (vector of 0s and K's).

    """

    if ftype.lower() == 'lowpass':
        pass_edge, stop_edge = edges
        pass_width = pass_edge
        stop_width = fs/2 - stop_edge
        num_pass = int(num_points * pass_width / (pass_width + stop_width))
        num_stop = num_points - num_pass

        omega_pass_norm = np.linspace(0, pass_edge, num_pass)
        omega_stop_norm = np.linspace(stop_edge, fs/2, num_stop)
        omega_grid_norm = np.concatenate((omega_pass_norm, omega_stop_norm))
        d_desired = np.concatenate((np.ones(num_pass)*K, np.zeros(num_stop)))
    elif ftype.lower() == 'highpass':
        stop_edge, pass_edge = edges
        stop_width = stop_edge
        pass_width = fs/2 - pass_edge
        num_stop = int(num_points * stop_width / (stop_width + pass_width))
        num_pass = num_points - num_stop

        omega_stop_norm = np.linspace(0, stop_edge, num_stop)
        omega_pass_norm = np.linspace(pass_edge, fs/2, num_pass)
        omega_grid_norm = np.concatenate((omega_stop_norm, omega_pass_norm))
        d_desired = np.concatenate((np.zeros(num_stop), np.ones(num_pass)*K))
    elif ftype.lower() == 'bandpass':
        stop1, pass1, pass2, stop2 = edges
        band1_width = stop1
        band2_width = pass2 - pass1
        band3_width = fs/2 - stop2
        total_width = band1_width + band2_width + band3_width

        num_band1 = int(num_points * band1_width / total_width)
        num_band2 = int(num_points * band2_width / total_width)
        num_band3 = num_points - num_band1 - num_band2

        omega_band1_norm = np.linspace(0, stop1, num_band1)
        omega_band2_norm = np.linspace(pass1, pass2, num_band2)
        omega_band3_norm = np.linspace(stop2, fs/2, num_band3)
        omega_grid_norm = np.concatenate((omega_band1_norm, omega_band2_norm, omega_band3_norm))
        d_desired = np.concatenate((
            np.zeros(num_band1),
            np.ones(num_band2)*K,
            np.zeros(num_band3)
        ))
    elif ftype.lower() in ['notch', 'bandstop']:
        pass1, stop1, stop2, pass2 = edges
        band1_width = pass1
        band2_width = stop2 - stop1
        band3_width = fs/2 - pass2
        total_width = band1_width + band2_width + band3_width

        num_band1 = int(num_points * band1_width / total_width)
        num_band2 = int(num_points * band2_width / total_width)
        num_band3 = num_points - num_band1 - num_band2

        omega_band1_norm = np.linspace(0, pass1, num_band1)
        omega_band2_norm = np.linspace(stop1, stop2, num_band2)
        omega_band3_norm = np.linspace(pass2, fs/2, num_band3)
        omega_grid_norm = np.concatenate((omega_band1_norm, omega_band2_norm, omega_band3_norm))
        d_desired = np.concatenate((
            np.ones(num_band1)*K,
            np.zeros(num_band2),
            np.ones(num_band3)*K
        ))
    else:
        raise ValueError(f"Unknown filter type '{ftype}'")

    angular_omega_grid = omega_grid_norm * np.pi
    return angular_omega_grid, d_desired


def compute_error_with_freqz(h_full: np.ndarray, specs: dict) -> float:
    """
    Computes the infinity norm error by analyzing the dense frequency response from freqz,
    supporting lowpass, highpass, bandpass, and bandstop/notch filters as specified in `specs`.
    
    Returns:
        The maximum error in any pass or stop band.
    """
    fs = specs['fs']
    K = specs['K']
    w, H = freqz(h_full, a=[1], worN=16384, fs=fs)
    filter_type = specs['filter_type'].lower()

    # Pass/stop edge helpers (None if not in specs)
    pe  = specs.get('pass_edge', None)
    se  = specs.get('stop_edge', None)
    pe2 = specs.get('pass_edge2', None)
    se2 = specs.get('stop_edge2', None)

    errors = []
    if filter_type == 'lowpass':
        # Passband: [0, pass_edge], Stopband: [stop_edge, fs/2]
        pass_idx = np.where(w <= pe)
        stop_idx = np.where(w >= se)
        errors.append(np.max(np.abs(np.abs(H[pass_idx]) - K)))
        errors.append(np.max(np.abs(H[stop_idx])))
    elif filter_type == 'highpass':
        # Passband: [pass_edge, fs/2], Stopband: [0, stop_edge]
        pass_idx = np.where(w >= pe)
        stop_idx = np.where(w <= se)
        errors.append(np.max(np.abs(np.abs(H[pass_idx]) - K)))
        errors.append(np.max(np.abs(H[stop_idx])))
    elif filter_type == 'bandpass':
        # Passband: [pass_edge, pass_edge2], Stopbands: [0, stop_edge], [stop_edge2, fs/2]
        if None in (pe, se, pe2, se2):
            raise ValueError("For bandpass, specify 'pass_edge', 'stop_edge', 'pass_edge2', 'stop_edge2'.")
        pass_idx = np.where((w >= pe) & (w <= pe2))
        stop1_idx = np.where(w <= se)
        stop2_idx = np.where(w >= se2)
        errors.append(np.max(np.abs(np.abs(H[pass_idx]) - K)))
        errors.append(np.max(np.abs(H[stop1_idx])))
        errors.append(np.max(np.abs(H[stop2_idx])))
    elif filter_type in ['notch', 'bandstop']:
        # Stopband: [stop_edge, stop_edge2], Passbands: [0, pass_edge], [pass_edge2, fs/2]
        if None in (pe, se, pe2, se2):
            raise ValueError("For bandstop/notch, specify 'pass_edge', 'stop_edge', 'pass_edge2', 'stop_edge2'.")
        pass1_idx = np.where(w <= pe)
        pass2_idx = np.where(w >= pe2)
        stop_idx = np.where((w >= se) & (w <= se2))
        errors.append(np.max(np.abs(np.abs(H[pass1_idx]) - K)))
        errors.append(np.max(np.abs(np.abs(H[pass2_idx]) - K)))
        errors.append(np.max(np.abs(H[stop_idx])))
    else:
        raise ValueError(f"Unknown filter type '{filter_type}'")

    return max(errors)

def FindNearest(weights: torch.Tensor, p_bits: int, device: torch.device):
    min_k = -2 ** (p_bits-1)
    max_k = 2 ** (p_bits-1) - 1
    k_values = torch.arange(min_k, max_k + 1, device=device)  

    quantization_levels = k_values / (2 ** (p_bits-1))
    diffs = torch.abs(weights.unsqueeze(0) - quantization_levels.unsqueeze(1))
    indices = diffs.argmin(dim=0)
    wq = quantization_levels[indices]
    return wq, indices.int()

def random_quantised_coeffs(order, p_bits, seed=1234):
    rng = np.random.default_rng(seed)
    half_len = order // 2 + 1
    int_min = -2 ** (p_bits - 1)
    int_max = 2 ** (p_bits - 1) - 1
    h_int = rng.integers(int_min, int_max + 1, size=half_len)
    idx_min, idx_max = rng.choice(half_len, size=2, replace=False)
    h_int[idx_min] = int_min
    h_int[idx_max] = int_max
    h0 = h_int.astype(float) / (2 ** (p_bits - 1))
    return h0

def quantize_coefficients_to_S(h_ideal: np.ndarray, p_bits: int) -> np.ndarray:
    """
    Rounds ideal, floating-point coefficients to the discrete set S.
    """
    scale_factor = 2**(p_bits-1)
    int_max = 2**(p_bits-1) - 1
    int_min = -2**(p_bits-1)
    
    h_scaled = h_ideal * scale_factor
    h_rounded_int = np.round(h_scaled)
    h_clipped_int = np.clip(h_rounded_int, int_min, int_max)
    h_quantized = h_clipped_int / scale_factor
    
    return h_quantized

def alns_fir(A, d_desired, project_specs, h0=None):
    if h0 is None:
        h0 = random_quantised_coeffs(project_specs['order'], project_specs['p_bits'])
    device = 'cuda'
    nQuantized = project_specs['p_bits']
    response = A @ h0
    print('intial inf norm before nearset: ', np.max(np.abs(response - d_desired)) )
    

    x0 = torch.from_numpy(h0).to(device)
    nearWq, nearQ = FindNearest(x0 , nQuantized, device)
    response = A @ nearWq.cpu().numpy()
    print('intial inf norm: ', np.max(np.abs(response - d_desired)) )
    W = torch.from_numpy(A).to(device)
    y= torch.from_numpy(d_desired).to(device)

    alns_obj = ALNS(nearQ, x0, W, nQuantized, debug=False, use_gptq=False, use_fir=True)
    alns_obj.set_stopping_criteria(stopping_criteria=MaxIterations(ALNS_ITERS))
    alns_obj.set_LS_operator('W')
    alns_obj.set_B_k(y)
    alns_obj.set_torch_device(device)
    solution = alns_obj.solve()

    return solution.quantized_weights.detach().cpu().numpy(), solution
def compute_error_with_matrix(h_full: np.ndarray, A, d_desired, L) -> float:
    """Computes the infinity norm error using the matrix multiplication method."""
    h_half = h_full[:L // 2 + 1]
    response = A @ h_half
    
    return np.max(np.abs(response - d_desired))

def create_fir_matrix_A(L: int, angular_omega_grid: np.ndarray) -> np.ndarray:
    """
    This matrix is constructed based on the formula derived from a causal
    filter representation:
    A(w) = h(M) + sum_{n=0 to M-1} 2*h(n)*cos(w*(M-n)).
    """
    M = L // 2

    n_vector_reversed = np.arange(M, -1, -1)
    # The columns will be [cos(M*w), cos((M-1)*w), ..., cos(0*w)]
    A = np.cos(np.outer(angular_omega_grid, n_vector_reversed))
    A[:, :-1] *= 2.0

    return A
if __name__ == "__main__":
    ALNS_ITERS = 20

    L = project_specs['order']
    fs = project_specs['fs']
    K = project_specs['K']
    grid_density = project_specs['grid_density']
    num_points = L * grid_density


    edges = get_remez_edges_from_specs(project_specs)
    w, d_desired = make_fir_desired(num_points, project_specs['filter_type'], edges, fs=fs)
    A = create_fir_matrix_A(L=L, angular_omega_grid=w)

    print(f"Specifications: Order N={project_specs['order']}, Bits p={project_specs['p_bits']}")


    h_ideal =  design_ideal_remez_filter(
        order=project_specs['order'],
        ftype=project_specs['filter_type'],
        edges=edges,
        fs=project_specs.get('fs', 2.0),
        K=project_specs.get('K', 1.0),
        gd=project_specs.get('grid_density', 16)
    )


    h_rounded = quantize_coefficients_to_S(h_ideal, p_bits=project_specs['p_bits'])
    t2=time.time()
    error_ideal_matrix = compute_error_with_matrix(h_ideal, A, d_desired, L)
    error_ideal_freqz = compute_error_with_freqz(h_ideal, project_specs)
    print(f"\nIdeal Filter Error (Matrix A):   {error_ideal_matrix:.8f}")
    print(f"Ideal Filter Error (freqz):      {error_ideal_freqz:.8f}")

    error_rounded_matrix = compute_error_with_matrix(h_rounded, A, d_desired, L)
    error_rounded_freqz = compute_error_with_freqz(h_rounded, project_specs)
    print(f"\nRounded Filter Error (Matrix A): {error_rounded_matrix:.8f}")
    print(f"Rounded Filter Error (freqz):    {error_rounded_freqz:.8f}")



    hr = copy.deepcopy(h_ideal[:L // 2 + 1])
    hf, solution = alns_fir(A, d_desired, project_specs, hr)

    response = A @ hf

    print("||Ax - b|| :", np.max(np.abs(response - d_desired)))
    h_alns= np.concatenate([hf, hf[-2::-1]])
    print("ALNS Filter Error (freqz) : ", compute_error_with_freqz(h_alns, project_specs))