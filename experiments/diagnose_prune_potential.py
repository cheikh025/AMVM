"""If we prune candidates against the best-so-far maximum, how much work survives?"""
import sys
sys.path.insert(0, "src/GPU"); sys.path.insert(0, "experiments")
import torch
from bench_row_solve import load_instance
from RoundToNearest import FindNearest
from ALNS.State import State
from ALNS.local_search import LocalSearch

dev = torch.device("mps")
for inst, M in (("model_decoder_layers_0_self_attn_q_proj", 768), ("model_decoder_layers_0_fc2", 3072)):
    name, X, W = load_instance(f"experiments/data/{inst}.pt", 16384, dev)
    row = W[0].contiguous(); _, q0 = FindNearest(row, 3, dev)
    st = State(X, q0, row, X @ row, 3, num_partial=100, LS_op="S", torch_device=dev,
               acceptance_policy="linf_l2_nonincrease")
    ls = LocalSearch(st); ls.delta_q = 1
    t = st.objective_value; bound = t - 0.001
    screen = st.L_set[1][:100]
    q1 = max(range(st.num_levels - 1), key=lambda q: len(ls.sol_dict[q]) * len(ls.sol_dict[q + 1]))
    q2 = q1 + 1
    ii_all = torch.tensor(ls.sol_dict[q1], device=dev); jj_all = torch.tensor(ls.sol_dict[q2], device=dev)
    delta = st.quantization_levels[q2] - st.quantization_levels[q1]
    pd = X[screen][:, ii_all, None] - X[screen][:, None, jj_all]
    mask = torch.all((st.signedD_ks[screen, None, None] + delta * pd).abs() <= bound, dim=0)
    si, sj = torch.nonzero(mask, as_tuple=True)
    ii, jj = ii_all[si][:4000], jj_all[sj][:4000]
    print(f"\n{inst} M={M}: level pair ({q1},{q2}) {mask.numel()} screened -> {int(mask.sum())} survivors (sample {len(ii)})")
    if len(ii) == 0: continue

    resid = st.signedD_ks[:, None] + delta * (X[:, ii] - X[:, jj])   # (N, C)
    true_max = resid.abs().max(0).values
    best = true_max.min()
    print(f"  current objective t={t:.6f}  best candidate max={best:.6f}  "
          f"candidates under t: {int((true_max < bound).sum())}/{len(ii)}")

    running = torch.zeros(len(ii), device=dev)
    stage, start, work, alive = 256, 0, 0, torch.arange(len(ii), device=dev)
    while start < len(X) and len(alive):
        rows = slice(start, start + stage)
        running[alive] = torch.maximum(running[alive], resid[rows][:, alive].abs().max(0).values)
        work += (min(start + stage, len(X)) - start) * len(alive)
        start += stage; stage = min(stage * 2, 16384)
        alive = alive[running[alive] <= best]      # prune against the true optimum
    print(f"  pruning against the best candidate: work {work/1e6:.1f}M of "
          f"{len(ii)*len(X)/1e6:.1f}M row-candidate products = {work/(len(ii)*len(X)):.3f} of full")

    # cost split: max versus sum of squares
    import time
    def timed(fn, n=5):
        fn(); torch.mps.synchronize(); s=time.perf_counter()
        for _ in range(n): fn()
        torch.mps.synchronize(); return (time.perf_counter()-s)/n*1000
    sub = (ii[:1024], jj[:1024])
    def both():
        r = st.signedD_ks[:, None] + delta * (X[:, sub[0]] - X[:, sub[1]])
        return r.abs().max(0).values, r.square().sum(0)
    def only_max():
        r = st.signedD_ks[:, None] + delta * (X[:, sub[0]] - X[:, sub[1]])
        return r.abs().max(0).values
    print(f"  1024 candidates x {len(X)} rows: max+squares {timed(both):.1f} ms, max only {timed(only_max):.1f} ms")
