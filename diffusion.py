"""DDPM: kosinový rozvrh šumu, dopředný proces q_sample, trénovací ztráta, sampling DDPM a DDIM.

Spuštění jako skript vykreslí milník fáze 2 – jak sprity postupně mizí v šumu:
  python diffusion.py --out runs/forward_process.png
"""
import argparse
import math

import torch
import torch.nn.functional as F


def cosine_betas(T, s=0.008):
    """Kosinový rozvrh (Nichol & Dhariwal 2021): ᾱ_t = cos²((t/T + s)/(1 + s) · π/2)."""
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)


class Diffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.device = device
        # skaláry v float64 na CPU (sampling je čte po jednom), kopie ᾱ na zařízení pro trénink
        self.betas = cosine_betas(T)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)
        self.alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), self.alpha_bar[:-1]])
        self.posterior_var = self.betas * (1 - self.alpha_bar_prev) / (1 - self.alpha_bar)
        self.alpha_bar_dev = self.alpha_bar.float().to(device)

    def q_sample(self, x0, t, noise=None):
        """x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε"""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar_dev[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model, x0):
        """L = E ‖ε − ε_θ(x_t, t)‖²"""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    def _predict_x0(self, model, x, t):
        """Odhad šumu → odhad čistého obrázku (oříznutý do [-1, 1]) a k němu konzistentní ε."""
        a = self.alpha_bar[t].item()
        eps = model(x, torch.full((x.shape[0],), t, device=x.device, dtype=torch.long))
        x0 = ((x - math.sqrt(1 - a) * eps) / math.sqrt(a)).clamp(-1, 1)
        eps = (x - math.sqrt(a) * x0) / math.sqrt(1 - a)
        return x0, eps

    @torch.no_grad()
    def ddpm_sample(self, model, n, size=32, generator=None, progress=None):
        """Původní DDPM: všech T kroků, v každém se přičte náhodný šum."""
        x = torch.randn(n, 3, size, size, generator=generator).to(self.device)
        ts = range(self.T - 1, -1, -1)
        for t in (progress(ts) if progress else ts):
            x0, _ = self._predict_x0(model, x, t)
            ab, ab_prev = self.alpha_bar[t].item(), self.alpha_bar_prev[t].item()
            beta, alpha = self.betas[t].item(), self.alphas[t].item()
            # střední hodnota posterioru q(x_{t-1} | x_t, x_0)
            mean = (math.sqrt(ab_prev) * beta / (1 - ab)) * x0 + (math.sqrt(alpha) * (1 - ab_prev) / (1 - ab)) * x
            if t > 0:
                z = torch.randn(x.shape, generator=generator).to(self.device)
                x = mean + math.sqrt(self.posterior_var[t].item()) * z
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, n=None, steps=50, eta=0.0, churn_above=None, size=32, x=None,
                    generator=None, progress=None):
        """DDIM (Song et al. 2020): přeskakuje kroky; s eta=0 je deterministický (stejný šum → stejný sprite).

        churn_above: pro kroky t >= churn_above použij eta=1. Kosinový rozvrh má ᾱ_999 ≈ 2e-9, takže
        ε-model tam x_0 odhadne jen s obří chybou; ořez na [-1, 1] z ní udělá saturovaný šum, který
        deterministický DDIM nese až do konce. Čerstvý šum v prvních krocích tuhle chybu přepíše.
        """
        if x is None:
            x = torch.randn(n, 3, size, size, generator=generator)
        x = x.to(self.device)
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        it = range(len(ts))
        for i in (progress(it) if progress else it):
            t = ts[i]
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            ab = self.alpha_bar[t].item()
            ab_prev = self.alpha_bar[t_prev].item() if t_prev >= 0 else 1.0
            x0, eps = self._predict_x0(model, x, t)
            step_eta = 1.0 if churn_above is not None and t >= churn_above else eta
            sigma = step_eta * math.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
            x = math.sqrt(ab_prev) * x0 + math.sqrt(max(0.0, 1 - ab_prev - sigma ** 2)) * eps
            if sigma > 0:
                x = x + sigma * torch.randn(x.shape, generator=generator).to(self.device)
        return x


def main():
    from data import load_dataset, save_grid

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sprites.pt")
    p.add_argument("--out", default="runs/forward_process.png")
    args = p.parse_args()

    x, _ = load_dataset(args.data)
    g = torch.Generator().manual_seed(0)
    x0 = x[torch.randperm(len(x), generator=g)[:8]]
    diff = Diffusion()
    ts = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    noise = torch.randn(x0.shape, generator=g)
    # řádek = jeden sprite, sloupce = rostoucí t
    rows = [diff.q_sample(x0, torch.full((8,), t), noise) for t in ts]
    grid = torch.stack(rows, 1).reshape(-1, 3, 32, 32)
    save_grid(grid, args.out, nrow=len(ts))
    print("ᾱ_t pro t =", ts, ":", [round(diff.alpha_bar[t].item(), 4) for t in ts])
    print(f"uloženo do {args.out}")


if __name__ == "__main__":
    main()
