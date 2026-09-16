"""Render one table per mode across every variant under experiments/results."""
import json
import pathlib
import statistics
import sys

LAYERS = ["self_attn_q_proj", "fc1", "fc2"]


def load(label, mode):
    directory = pathlib.Path("experiments/results") / label
    values = {}
    for path in sorted(directory.glob(f"*_{mode}.json")):
        records = json.loads(path.read_text())["records"]
        layer = path.stem.replace("model_decoder_layers_0_", "").replace(f"_{mode}", "")
        values[layer] = records
    return values


def main():
    order = sys.argv[1:] or ["baseline", "a2-host-indexing", "a1-skip-settled",
                             "a3-device-resident", "a4-budget-tile", "a6-prune-incumbent"]
    for mode, metric, label in (("speed", "ms per iteration", "lower is better"),
                                ("quality", "final infinity norm", "lower is better")):
        present = [name for name in order
                   if (pathlib.Path("experiments/results") / name).exists()
                   and load(name, mode)]
        if not present:
            continue
        print(f"\n=== {mode}: {metric} ({label})")
        print(f"{'variant':22s}" + "".join(f"{layer:>18s}" for layer in LAYERS))
        reference = None
        for name in present:
            data = load(name, mode)
            cells, numbers = [], {}
            for layer in LAYERS:
                records = data.get(layer)
                if not records:
                    cells.append(f"{'-':>18s}")
                    continue
                if mode == "speed":
                    value = statistics.median(record["seconds"] / max(record["iterations"], 1)
                                              * 1000 for record in records)
                else:
                    value = statistics.median(record["final_objective"] for record in records)
                numbers[layer] = value
                if reference and layer in reference:
                    ratio = reference[layer] / value if mode == "speed" else value / reference[layer]
                    suffix = f" {ratio:5.2f}x" if mode == "speed" else f" {ratio:6.4f}"
                    cells.append(f"{value:11.4f}{suffix}" if mode == "quality"
                                 else f"{value:11.0f}{suffix}")
                else:
                    cells.append(f"{value:18.4f}" if mode == "quality" else f"{value:18.0f}")
            print(f"{name:22s}" + "".join(cells))
            if reference is None:
                reference = numbers
        print("  ratios are against the first row" if reference else "")


if __name__ == "__main__":
    main()
