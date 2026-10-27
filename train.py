"""Trénink DDPM na spritech: AdamW, EMA vah, checkpointy a mřížka vzorků každých N kroků.

  python train.py --out runs/main --steps 60000
  python train.py --out runs/overfit --overfit 64 --steps 3000   # sanity check: musí zreprodukovat 64 obrázků
  python train.py --out runs/main --resume                       # pokračovat z runs/main/ckpt.pt

Ctrl+C trénink přeruší a uloží checkpoint.
"""
import argparse
import copy
import csv
import os
import time

import torch
from tqdm import tqdm

from data import load_dataset, random_flip, save_grid
from diffusion import Diffusion
from unet import UNet, count_params


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def ema_update(ema, model, decay):
    ema_p = list(ema.parameters())
    torch._foreach_lerp_(ema_p, list(model.parameters()), 1 - decay)


def save_checkpoint(path, model, ema, opt, step, args):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "step": step, "args": vars(args)}, tmp)
    os.replace(tmp, path)  # atomicky, ať přerušení během ukládání nezničí poslední checkpoint


def plot_loss(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    plt.figure(figsize=(7, 3.5))
    plt.plot(steps, loss, lw=1)
    plt.yscale("log")
    plt.xlabel("krok")
    plt.ylabel("MSE loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sprites.pt")
    p.add_argument("--out", default="runs/main")
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--overfit", type=int, default=0, help="trénovat jen na prvních N obrázcích")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    data, _ = load_dataset(args.data, device)
    if args.overfit:
        data = data[:args.overfit]
        save_grid(data[:64], os.path.join(args.out, "overfit_targets.png"))
    n = len(data)

    model = UNet(dropout=0.0 if args.overfit else args.dropout).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    diff = Diffusion(device=device)
    ckpt_path = os.path.join(args.out, "ckpt.pt")
    log_path = os.path.join(args.out, "loss.csv")

    step = 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"pokračuji od kroku {step}")
    elif not args.resume or not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("step,loss,sec_per_step\n")

    print(f"zařízení: {device}, dat: {n}, parametrů: {count_params(model) / 1e6:.2f} M")
    # pevný šum pro vzorky, aby šlo porovnávat pokrok mezi checkpointy
    fixed_noise = torch.randn(64, 3, 32, 32, generator=torch.Generator().manual_seed(1234))

    model.train()
    loss_acc = torch.zeros((), device=device)
    t_last = time.time()
    pbar = tqdm(total=args.steps, initial=step, dynamic_ncols=True)
    try:
        while step < args.steps:
            idx = torch.randint(0, n, (args.batch,), device=device)
            x0 = data[idx] if args.overfit else random_flip(data[idx])

            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
            loss = diff.loss(model, x0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # na začátku kratší průměrování, ať EMA nestojí na náhodné inicializaci
            ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))
            loss_acc += loss.detach()
            step += 1
            pbar.update(1)

            if step % args.log_every == 0:
                avg = loss_acc.item() / args.log_every  # .item() synchronizuje GPU, proto jen občas
                loss_acc.zero_()
                sps = (time.time() - t_last) / args.log_every
                t_last = time.time()
                pbar.set_postfix(loss=f"{avg:.4f}", s_step=f"{sps:.3f}")
                with open(log_path, "a") as f:
                    f.write(f"{step},{avg:.6f},{sps:.4f}\n")

            if step % args.sample_every == 0 or step == args.steps:
                # deterministický DDIM (eta=0) i stochastický (eta=1), ať je vidět rozdíl během tréninku
                for eta in (0.0, 1.0):
                    samples = diff.ddim_sample(ema, x=fixed_noise, steps=50, eta=eta,
                                               generator=torch.Generator().manual_seed(0))
                    save_grid(samples, os.path.join(args.out, "samples", f"step_{step:06d}_eta{eta:g}.png"))
                save_checkpoint(ckpt_path, model, ema, opt, step, args)
                plot_loss(log_path, os.path.join(args.out, "loss.png"))
                t_last = time.time()  # sampling nezapočítávat do rychlosti tréninku
    except KeyboardInterrupt:
        print("\npřerušeno, ukládám checkpoint")
    finally:
        pbar.close()
        save_checkpoint(ckpt_path, model, ema, opt, step, args)
        torch.save({"ema": ema.state_dict(), "step": step}, os.path.join(args.out, "ema.pt"))
        print(f"krok {step}, checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
