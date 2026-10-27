"""Generování spritů z natrénovaného modelu (EMA váhy).

  python sample.py --ckpt runs/main/ema.pt                      # mřížka 64 spritů, DDIM 50 kroků
  python sample.py --ckpt runs/main/ema.pt --churn-above -1     # čistý deterministický DDIM (viz diffusion.py)
  python sample.py --ckpt runs/main/ema.pt --sampler ddpm       # původní DDPM, 1000 kroků
  python sample.py --ckpt runs/main/ema.pt --quantize 12        # + paleta 12 barev pro každý sprite
  python sample.py --ckpt runs/main/ema.pt --quantize 32 --palette dataset   # + společná paleta z datasetu
  python sample.py --ckpt runs/main/ema.pt --nn                 # + kontrola kopií (nejbližší soused v datasetu)
  python sample.py --ckpt runs/main/ema.pt --interp             # + interpolace mezi dvěma sprity
"""
import argparse
import math
import os
import time

import torch
from tqdm import tqdm

from data import load_dataset, save_grid
from diffusion import Diffusion
from train import get_device
from unet import UNet


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = UNet().to(device).eval()
    model.load_state_dict(ck["ema"])
    return model, ck.get("step")


def kmeans_palette(x, k, iters=25, n_pixels=200_000, seed=0):
    """Paleta k barev z pixelů datasetu (x v [-1,1], [N,3,H,W]); jednoduchý k-means."""
    g = torch.Generator().manual_seed(seed)
    px = x.permute(0, 2, 3, 1).reshape(-1, 3)
    px = px[torch.randint(0, len(px), (n_pixels,), generator=g)]
    centers = px[torch.randperm(len(px), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(px, centers).argmin(1)
        for j in range(k):
            members = px[assign == j]
            if len(members):
                centers[j] = members.mean(0)
    return centers


def quantize_per_sprite(x, k):
    """Každý sprite dostane vlastní paletu k barev (k-means jen z jeho pixelů)."""
    return torch.cat([quantize(xi[None], kmeans_palette(xi[None], k, n_pixels=xi[0].numel())) for xi in x])


def quantize(x, palette):
    """Každý pixel nahradí nejbližší barvou z palety."""
    B, C, H, W = x.shape
    px = x.permute(0, 2, 3, 1).reshape(-1, 3)
    q = palette[torch.cdist(px, palette).argmin(1)]
    return q.reshape(B, H, W, C).permute(0, 3, 1, 2)


def nearest_neighbors(samples, data, chunk=4096):
    """L2 nejbližší soused každého vzorku v datasetu (včetně zrcadlených verzí)."""
    s = samples.reshape(len(samples), -1)
    best_d = torch.full((len(samples),), float("inf"))
    best_i = torch.zeros(len(samples), dtype=torch.long)
    for flip in (False, True):
        d_all = data.flip(3) if flip else data
        for i in range(0, len(d_all), chunk):
            d = torch.cdist(s, d_all[i:i + chunk].reshape(-1, s.shape[1]))
            m, j = d.min(1)
            better = m < best_d
            best_d[better] = m[better]
            best_i[better] = (j + i)[better]
    return best_d, best_i


def slerp(a, b, w):
    """Sférická interpolace šumu: zachová normu, takže mezikroky jsou pro model „rozumný“ šum."""
    af, bf = a.flatten(), b.flatten()
    omega = torch.acos((af @ bf / (af.norm() * bf.norm())).clamp(-1, 1))
    return (torch.sin((1 - w) * omega) * a + torch.sin(w * omega) * b) / torch.sin(omega)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/main/ema.pt")
    p.add_argument("--data", default="data/sprites.pt")
    p.add_argument("--out", default=None, help="výchozí: <složka checkpointu>/generated")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=50, help="počet kroků DDIM")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--churn-above", type=int, default=700,
                   help="kroky s t >= této hodnoty používají eta=1 (-1 = vypnuto, čistý DDIM)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantize", type=int, default=0, help="velikost palety (0 = vypnuto)")
    p.add_argument("--palette", choices=["sprite", "dataset"], default="sprite",
                   help="paleta pro každý sprite zvlášť, nebo jedna společná z datasetu")
    p.add_argument("--nn", action="store_true")
    p.add_argument("--interp", action="store_true")
    p.add_argument("--device", default=None, help="výchozí: mps, pokud je k dispozici")
    args = p.parse_args()

    device = args.device or get_device()
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "generated")
    os.makedirs(out, exist_ok=True)
    model, step = load_model(args.ckpt, device)
    diff = Diffusion(device=device)
    g = torch.Generator().manual_seed(args.seed)
    churn = None if args.churn_above < 0 else args.churn_above
    print(f"model z kroku {step}, zařízení {device}")

    t0 = time.time()
    if args.sampler == "ddim":
        x = diff.ddim_sample(model, args.n, steps=args.steps, eta=args.eta, churn_above=churn,
                             generator=g, progress=tqdm)
    else:
        x = diff.ddpm_sample(model, args.n, generator=g, progress=tqdm)
    x = x.cpu()
    elapsed = time.time() - t0
    name = f"ddim{args.steps}_eta{args.eta:g}" if args.sampler == "ddim" else "ddpm"
    if args.sampler == "ddim" and churn is not None:
        name += f"_churn{churn}"
    name += f"_seed{args.seed}"
    save_grid(x, os.path.join(out, f"{name}.png"))
    print(f"{args.n} spritů za {elapsed:.1f} s -> {out}/{name}.png")

    need_data = (args.quantize and args.palette == "dataset") or args.nn
    data = load_dataset(args.data)[0] if need_data else None

    if args.quantize:
        qname = f"{name}_q{args.quantize}{args.palette}"
        if args.palette == "dataset":
            palette = kmeans_palette(data, args.quantize)
            xq = quantize(x, palette)
            save_grid(palette.T.reshape(1, 3, 1, args.quantize), os.path.join(out, f"palette_{args.quantize}.png"),
                      scale=16)
        else:
            xq = quantize_per_sprite(x, args.quantize)
        save_grid(xq, os.path.join(out, f"{qname}.png"))
        print(f"kvantizováno na {args.quantize} barev ({args.palette}) -> {out}/{qname}.png")

    if args.nn:
        k = min(16, args.n)
        d, idx = nearest_neighbors(x[:k], data)
        # referenční vzdálenost: jak daleko mají k sobě nejbližší dvojice uvnitř datasetu
        ref_d, _ = nearest_neighbors(data[:200], data[200:])
        pairs = torch.stack([x[:k], data[idx]], 1).reshape(-1, 3, 32, 32)
        save_grid(pairs, os.path.join(out, f"{name}_nn.png"), nrow=8)
        print("L2 vzorek→nejbližší v datasetu:", " ".join(f"{v:.1f}" for v in d.tolist()))
        print(f"medián vzorky: {d.median():.1f}, medián dataset→dataset: {ref_d.median():.1f} "
              f"(minimum {ref_d.min():.1f}); dvojice uloženy do {out}/{name}_nn.png")

    if args.interp:
        rows = []
        for r in range(4):
            a = torch.randn(1, 3, 32, 32, generator=g)
            b = torch.randn(1, 3, 32, 32, generator=g)
            rows.append(torch.cat([slerp(a, b, w) for w in torch.linspace(0, 1, 8).tolist()]))
        z = torch.cat(rows)
        # interpolace potřebuje čistý deterministický DDIM: s churn_above by šum v prvních krocích x_T přepsal
        xi = diff.ddim_sample(model, x=z, steps=args.steps).cpu()
        save_grid(xi, os.path.join(out, f"interp_seed{args.seed}.png"), nrow=8)
        print(f"interpolace -> {out}/interp_seed{args.seed}.png")


if __name__ == "__main__":
    main()
