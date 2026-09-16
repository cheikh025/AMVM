"""Compare dense and tiled swap evaluation on identical synthetic inputs."""

import argparse
import json
import statistics
import time

import torch


def dense(inputs, residual, left, right, delta, screen_rows, epsilon):
    differences = inputs[screen_rows][:, left, None] - inputs[screen_rows][:, None, right]
    mask = torch.all((residual[screen_rows, None, None] + delta * differences).abs()
                     < residual.abs().max() - epsilon, dim=0)
    ii, jj = torch.nonzero(mask, as_tuple=True)
    if len(ii) == 0:
        return None
    candidates = residual[:, None] + delta * (inputs[:, left[ii]] - inputs[:, right[jj]])
    linf = candidates.abs().max(dim=0).values
    l2 = candidates.square().sum(dim=0)
    best_linf = linf.min()
    tied = torch.nonzero(linf == best_linf, as_tuple=True)[0]
    chosen = tied[torch.argmin(l2[tied])]
    return best_linf.item(), l2[chosen].item(), int(left[ii[chosen]]), int(right[jj[chosen]])


def tiled(inputs, residual, left, right, delta, screen_rows, epsilon,
          variable_tile, row_tile, candidate_chunk):
    best = None
    threshold = residual.abs().max() - epsilon
    for i_start in range(0, len(left), variable_tile):
        tile_i = left[i_start:i_start + variable_tile]
        for j_start in range(0, len(right), variable_tile):
            tile_j = right[j_start:j_start + variable_tile]
            mask = torch.ones((len(tile_i), len(tile_j)), dtype=torch.bool,
                              device=inputs.device)
            for row_start in range(0, len(screen_rows), row_tile):
                rows = screen_rows[row_start:row_start + row_tile]
                differences = inputs[rows][:, tile_i, None] - inputs[rows][:, None, tile_j]
                mask &= torch.all((residual[rows, None, None] + delta * differences).abs()
                                  < threshold, dim=0)
            ii, jj = torch.nonzero(mask, as_tuple=True)
            for start in range(0, len(ii), candidate_chunk):
                candidate_i = tile_i[ii[start:start + candidate_chunk]]
                candidate_j = tile_j[jj[start:start + candidate_chunk]]
                maxima = torch.zeros(len(candidate_i), device=inputs.device)
                squares = torch.zeros_like(maxima)
                for row_start in range(0, len(inputs), row_tile):
                    rows = slice(row_start, row_start + row_tile)
                    values = (residual[rows, None] + delta
                              * (inputs[rows, candidate_i] - inputs[rows, candidate_j]))
                    maxima = torch.maximum(maxima, values.abs().max(dim=0).values)
                    squares += values.square().sum(dim=0)
                if len(maxima) == 0:
                    continue
                minimum = maxima.min()
                tied = torch.nonzero(maxima == minimum, as_tuple=True)[0]
                chosen = tied[torch.argmin(squares[tied])]
                proposal = (minimum.item(), squares[chosen].item(),
                            int(candidate_i[chosen]), int(candidate_j[chosen]))
                if best is None or proposal[:2] < best[:2]:
                    best = proposal
    return best


def timed(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--screen-rows", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=512)
    parser.add_argument("--variable-tile", type=int, default=64)
    parser.add_argument("--row-tile", type=int, default=32)
    parser.add_argument("--candidate-chunk", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(9101)
    columns = 2 * args.group_size
    inputs = torch.randn(args.rows, columns, generator=generator, device=device)
    residual = 4 * torch.randn(args.rows, generator=generator, device=device)
    left = torch.arange(args.group_size, device=device)
    right = torch.arange(args.group_size, columns, device=device)
    screen_rows = residual.abs().topk(min(args.screen_rows, args.rows)).indices
    delta, epsilon = 0.125, 0.001

    dense_result, dense_seconds = timed(
        lambda: dense(inputs, residual, left, right, delta, screen_rows, epsilon),
        args.repeats)
    tiled_result, tiled_seconds = timed(
        lambda: tiled(inputs, residual, left, right, delta, screen_rows, epsilon,
                      args.variable_tile, args.row_tile, args.candidate_chunk),
        args.repeats)
    if dense_result != tiled_result:
        raise AssertionError(f"dense={dense_result}, tiled={tiled_result}")

    element_bytes = inputs.element_size()
    dense_filter_bytes = args.screen_rows * args.group_size ** 2 * element_bytes
    tiled_filter_bytes = (min(args.row_tile, args.screen_rows)
                          * min(args.variable_tile, args.group_size) ** 2 * element_bytes)
    report = {
        "device": str(device),
        "shape": {"rows": args.rows, "group_size": args.group_size,
                  "screen_rows": args.screen_rows},
        "best": dense_result,
        "dense_median_seconds": dense_seconds,
        "tiled_median_seconds": tiled_seconds,
        "speed_ratio_dense_over_tiled": dense_seconds / tiled_seconds,
        "dense_filter_tensor_bytes": dense_filter_bytes,
        "tiled_filter_tensor_bytes_upper_bound": tiled_filter_bytes,
        "filter_tensor_reduction": dense_filter_bytes / tiled_filter_bytes,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
