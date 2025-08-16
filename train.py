import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from orpheus.orpheus import Orpheus
from src.orpheus.data.data import TinyShakeDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Orpheus")
    parser.add_argument(
        "--resume-from",
        "-r",
        dest="resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    start_step = 0

    # Attempt to read config from checkpoint if resuming
    ckpt_cfg = None
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"Resuming from checkpoint: {ckpt_path}")
        tmp = torch.load(ckpt_path, map_location="cpu")
        if isinstance(tmp, dict) and "config" in tmp:
            ckpt_cfg = tmp["config"]
            start_step = int(tmp.get("step", 0))

    model = Orpheus(
        block_size=(ckpt_cfg.get("block_size") if ckpt_cfg else 256),
        vocab_size=(ckpt_cfg.get("vocab_size") if ckpt_cfg else 65),
        n_embed=(ckpt_cfg.get("n_embed") if ckpt_cfg else 380),
        dropout=(ckpt_cfg.get("dropout") if ckpt_cfg else 0.2),
    ).to(device)

    # Load model/optimizer state if resuming
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])  # type: ignore[arg-type]
            if "optimizer_state" in state:
                model.optimiser.load_state_dict(state["optimizer_state"])  # type: ignore[arg-type]
        else:
            # Raw state_dict only
            model.load_state_dict(state)  # type: ignore[arg-type]
        print(f"Loaded checkpoint state. Starting at step {start_step}.")

    data_module = TinyShakeDataModule(
        data_path="/Users/tobyhallett/Desktop/orpheus/resources/tiny_shakespeare.txt",
        block_size=256,
        batch_size=2,
    )

    train_loader = data_module.train_dataloader()

    model.train()
    for step, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(xb, yb)
        model.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        model.optimiser.step()

        global_step = start_step + step
        if global_step % 50 == 0:
            print(f"step {global_step}: loss={loss.item():.4f}")
        if step >= 2000:
            break

    # Save final checkpoint
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "orpheus_final.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": model.optimiser.state_dict(),
            "config": {
                "block_size": model.block_size,
                "n_embed": model.n_embed,
                "dropout": model.dropout,
                "vocab_size": model.token_embedding_table.num_embeddings,
            },
            "step": global_step,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    # generate some sample text with the model
    seed = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(seed, max_new_tokens=1000)[0].tolist()
    decoder = data_module.int_to_char # dictionary mapping integers to characters

    # Decode ints -> chars (will raise fast if an id is missing)
    try:
        text = "".join(decoder[i] for i in generated)
    except KeyError as e:
        raise ValueError(f"Unknown token id in output: {e.args[0]} (vocab size={len(decoder)})")

    # Save to file
    out_path = Path("/Users/tobyhallett/Desktop/orpheus/outputs/output1.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"Saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
