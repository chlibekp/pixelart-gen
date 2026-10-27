"""Malý U-Net pro odhad šumu ε_θ(x_t, t).

Tři úrovně rozlišení 32→16→8 px s kanály 64/128/256, ResNet bloky s GroupNorm a SiLU,
sinusové kódování času přičtené do každého ResNet bloku, self-attention jen na 8×8.

Spuštění jako skript zkontroluje tvary a vypíše počet parametrů (milník fáze 3):
  python unet.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusové kódování kroku t (jako poziční kódování v Transformeru)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(32, cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # blok na začátku tréninku ≈ identita
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        # [B, heads, HW, d]
        q, k, v = (z.transpose(-1, -2) for z in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class Block(nn.Module):
    """ResNet blok, volitelně následovaný self-attention."""

    def __init__(self, cin, cout, tdim, dropout, attn):
        super().__init__()
        self.res = ResBlock(cin, cout, tdim, dropout)
        self.attn = Attention(cout) if attn else nn.Identity()

    def forward(self, x, temb):
        return self.attn(self.res(x, temb))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 256), num_res=2, attn_res=(8,), img_size=32, img_ch=3, dropout=0.1):
        super().__init__()
        self.base = channels[0]
        tdim = channels[0] * 4
        self.time_mlp = nn.Sequential(nn.Linear(self.base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(img_ch, self.base, 3, padding=1)

        # sestupná cesta; skip_ch si pamatuje kanály všech výstupů pro skip spojení
        self.down = nn.ModuleList()
        ch, res = self.base, img_size
        skip_ch = [ch]
        for i, cout in enumerate(channels):
            for _ in range(num_res):
                self.down.append(Block(ch, cout, tdim, dropout, res in attn_res))
                ch = cout
                skip_ch.append(ch)
            if i < len(channels) - 1:
                self.down.append(Downsample(ch))
                res //= 2
                skip_ch.append(ch)

        self.mid = nn.ModuleList([Block(ch, ch, tdim, dropout, True), Block(ch, ch, tdim, dropout, False)])

        # vzestupná cesta; každý blok spojí aktuální stav s jedním skip výstupem
        self.up = nn.ModuleList()
        for i, cout in reversed(list(enumerate(channels))):
            for _ in range(num_res + 1):
                self.up.append(Block(ch + skip_ch.pop(), cout, tdim, dropout, res in attn_res))
                ch = cout
            if i > 0:
                self.up.append(Upsample(ch))
                res *= 2

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, img_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.conv_in(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, Block) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, temb)
        for m in self.up:
            h = m(torch.cat([h, hs.pop()], 1), temb) if isinstance(m, Block) else m(h)
        return self.out(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import time

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNet().to(device)
    x = torch.randn(128, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (128,), device=device)
    y = model(x, t)
    assert y.shape == x.shape, y.shape
    print(f"zařízení: {device}")
    print(f"vstup {tuple(x.shape)} -> výstup {tuple(y.shape)}")
    print(f"parametrů: {count_params(model) / 1e6:.2f} M")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for i in range(13):  # rychlý odhad rychlosti tréninkového kroku
        if i == 3:
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.time()
        loss = (model(x, t) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()
    print(f"trénovací krok (batch 128): {(time.time() - t0) / 10 * 1000:.0f} ms")
