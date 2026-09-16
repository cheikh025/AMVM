"""Capture real opt-125m weight rows and layer activations as benchmark instances.

Produces one .pt per layer under ``experiments/data`` holding a subsampled
activation matrix ``X`` (n_rows x M) and the full weight matrix ``W`` (R x M).
The activation rows are subsampled with a fixed seed so runs are reproducible
and the files stay small enough to load on a laptop.

The files are inputs to ``experiments/bench_row_solve.py``; they are not checked
into git (see .gitignore). Re-run this script to regenerate them.
"""
import argparse
import pathlib

import torch


def calibration_texts(n_sequences):
    """Return calibration strings, preferring wikitext2 over a local fallback."""
    try:
        from datasets import load_dataset
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        joined = "\n\n".join(data["text"])
        return [joined], "wikitext2"
    except Exception as error:  # pragma: no cover - network/dependency dependent
        print(f"wikitext2 unavailable ({type(error).__name__}: {error}); using repository text")
        root = pathlib.Path(__file__).resolve().parents[1]
        chunks = []
        for path in sorted(root.rglob("*.py"))[:120]:
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
        return ["\n".join(chunks)], "repository-source-fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--layers", nargs="+",
                        default=["model.decoder.layers.0.self_attn.q_proj",
                                 "model.decoder.layers.0.fc1",
                                 "model.decoder.layers.0.fc2"])
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--keep-rows", type=int, default=65536,
                        help="activation rows retained per layer after subsampling")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--out-dir", default="experiments/data")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    texts, source = calibration_texts(args.sequences)
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")
    total = encoded.input_ids.shape[1]
    usable = min(args.sequences, max(1, total // args.seq_len))
    print(f"calibration source {source}: {total} tokens, using {usable} sequences")

    captured = {name: [] for name in args.layers}
    handles = []
    for name, module in model.named_modules():
        if name in captured:
            def hook(mod, inputs, output, key=name):
                captured[key].append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]))
            handles.append(module.register_forward_hook(hook))
    missing = [name for name in args.layers if not any(n == name for n, _ in model.named_modules())]
    if missing:
        raise SystemExit(f"layers not found in {args.model}: {missing}")

    with torch.no_grad():
        for index in range(usable):
            start = index * args.seq_len
            ids = encoded.input_ids[:, start:start + args.seq_len]
            model(ids)
            print(f"captured sequence {index + 1}/{usable}")

    for handle in handles:
        handle.remove()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    weights = dict(model.named_modules())
    for name in args.layers:
        X = torch.cat(captured[name], dim=0)
        if X.shape[0] > args.keep_rows:
            rows = torch.randperm(X.shape[0], generator=generator)[:args.keep_rows].sort().values
            X = X[rows]
        W = weights[name].weight.detach().clone()
        path = out_dir / f"{name.replace('.', '_')}.pt"
        torch.save({"name": name, "X": X.contiguous(), "W": W.contiguous(),
                    "source": source, "sequences": usable, "seq_len": args.seq_len,
                    "keep_rows": args.keep_rows, "seed": args.seed}, path)
        print(f"saved {path}  X={tuple(X.shape)}  W={tuple(W.shape)}")


if __name__ == "__main__":
    main()
