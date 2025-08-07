import gurobipy as gp
from gurobipy import GRB
import numpy as np
from fir_config import project_specs
def get_remez_edges_from_specs(specs):
    ftype = specs['filter_type'].lower()
    pe = specs.get('pass_edge', None)
    se = specs.get('stop_edge', None)
    pe2 = specs.get('pass_edge2', None)
    se2 = specs.get('stop_edge2', None)
    fs = specs.get('fs', 2.0)
    if ftype == 'lowpass':
        edges = (pe, se)
    elif ftype == 'highpass':
        edges = (se, pe)
    elif ftype == 'bandpass':
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

def make_fir_desired(num_points, ftype, edges, fs=2.0, K=1.0,):

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


def design_fir_discrete(A: np.ndarray,
                        d_desired: np.ndarray,
                        p_bits: int,
                        mip_gap: float = 0.0,
                        time_limit: float = None):

    m_rows, n_vars = A.shape
    two_p = 2 ** (p_bits-1)           
    lb, ub = -2**(p_bits-1), 2**(p_bits-1) - 1  

   
    model = gp.Model('FIR_discrete')
    model.Params.OutputFlag = 0     
    model.Params.MIPGap = mip_gap
    
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    x = model.addVars(n_vars, lb=lb, ub=ub, vtype=GRB.INTEGER,
                      name='x')                      
    delta = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='Delta')  

   
    for j in range(m_rows):

        lhs = gp.quicksum(A[j, k] * x[k] for k in range(n_vars))
        model.addConstr( lhs - two_p * d_desired[j] <= delta, name=f'ub_{j}')
        model.addConstr(-lhs + two_p * d_desired[j] <= delta, name=f'lb_{j}')

    model.setObjective(delta, GRB.MINIMIZE)
    model.optimize()
    status   = model.Status
    sols     = model.SolCount
    obj_val  = model.ObjVal    if sols > 0 else None
    obj_bound= model.ObjBound
    if status == GRB.Status.OPTIMAL:
        print(f"Optimal! Obj = {obj_val:.4f}")
    elif status == GRB.Status.TIME_LIMIT:
        if sols > 0:
            print(f" Time limit reached. Best feasible obj = {obj_val:.4f}")
            print(f"              Bound = {obj_bound:.4f}, Gap = {model.MIPGap:.2%}")
        else:
            print("Time limit reached. No feasible solution found.")
    else:
        print(f"Solver stopped with status {status}")
        print(f"Optimization completed in {model.Runtime:.2f} seconds.")
  
    coeff_int = np.array([x[k].X for k in range(n_vars)], dtype=int)
    coeff = coeff_int / two_p
    ripple = delta.X / two_p        

    return coeff, ripple, model



    
L = project_specs['order']
fs = project_specs['fs']
K = project_specs['K']
grid_density = project_specs['grid_density']
num_points = L * grid_density
p_bits = project_specs['p_bits']

device = 'cuda'
nQuantized = project_specs['p_bits']

edges = get_remez_edges_from_specs(project_specs)
angular_omega_grid, d_desired = make_fir_desired(num_points, project_specs['filter_type'], edges, fs=fs)
A = create_fir_matrix_A(L=L, angular_omega_grid=angular_omega_grid)


print(A.shape)

h_opt, ripple, mdl = design_fir_discrete(A, d_desired, p_bits=p_bits,  time_limit=120)
print('Gurobi Filter Error =', ripple, '\nTaps  =', h_opt)
