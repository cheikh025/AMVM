"""How many rows does it take to kill a screening survivor, under different row orders?"""
import sys, pathlib
sys.path.insert(0, "src/GPU"); sys.path.insert(0, "experiments")
import torch
from bench_row_solve import load_instance
from RoundToNearest import FindNearest
from ALNS.State import State
from ALNS.local_search import LocalSearch, NUM_FILTERS

dev = torch.device("mps")
name, X, W = load_instance("experiments/data/model_decoder_layers_0_self_attn_q_proj.pt", 16384, dev)
row = W[0].contiguous(); _, q0 = FindNearest(row, 3, dev)
st = State(X, q0, row, X @ row, 3, num_partial=100, LS_op="S", torch_device=dev,
           acceptance_policy="linf_l2_nonincrease")
ls = LocalSearch(st); ls.delta_q = 1
t = st.objective_value; bound = t - 0.001
screen = st.L_set[1][:100]

# Reproduce one level pair's survivors
q1, q2 = 3, 4
ii_all = torch.tensor(ls.sol_dict[q1], device=dev); jj_all = torch.tensor(ls.sol_dict[q2], device=dev)
delta = st.quantization_levels[q2] - st.quantization_levels[q1]
pd = X[screen][:, ii_all, None] - X[screen][:, None, jj_all]
mask = torch.all((st.signedD_ks[screen, None, None] + delta * pd).abs() <= bound, dim=0)
si, sj = torch.nonzero(mask, as_tuple=True)
ii, jj = ii_all[si], jj_all[sj]
print(f"level pair ({q1},{q2}): {mask.numel()} pairs screened, {len(ii)} survivors")
if len(ii) == 0: sys.exit()
ii, jj = ii[:2000], jj[:2000]

diff = X[:, ii] - X[:, jj]                      # (N, C)
resid = st.signedD_ks[:, None] + delta * diff   # (N, C)
over = resid.abs() > bound                      # which rows kill which candidate
print(f"candidates killed by some row: {int(over.any(0).sum())} of {len(ii)}")

orders = {
  "residual |D_k| desc": st.absD_ks.argsort(descending=True),
  "row norm ||x_k|| desc": X.norm(dim=1).argsort(descending=True),
  "row max|x| desc": X.abs().max(dim=1).values.argsort(descending=True),
  "natural": torch.arange(len(X), device=dev),
  "spread of the two columns desc": (X[:, ii] - X[:, jj]).abs().max(dim=1).values.argsort(descending=True),
}
for label, order in orders.items():
    ordered = over[order]                        # (N, C) in this row order
    first = torch.where(ordered.any(0), ordered.float().argmax(0), torch.tensor(len(X)-1, device=dev))
    q = torch.tensor([0.5, 0.9, 0.99], device=dev)
    quant = torch.quantile(first.float(), q).tolist()
    print(f"{label:32s} rows to kill: median {quant[0]:8.0f}  p90 {quant[1]:8.0f}  p99 {quant[2]:8.0f}"
          f"   work vs full: {float(first.float().mean())/len(X):.3f}")
