# Zadání: Generativní model pixel artu

Oct 26, 2026 · @Petr Chlíbek

## Cíl projektu

Napsat od nuly a natrénovat na MacBooku Air M2 malý difuzní model, který generuje nové 32×32 pixel-art postavičky (sprity). Celý kód musí být vlastní a srozumitelný, bez hotových pipeline z knihoven typu diffusers.

Výstupy projektu:

- Repozitář s kódem: příprava dat, model, trénink, generování.
- Natrénovaný checkpoint, který na laptopu vygeneruje mřížku 64 spritů za méně než minutu.
- Krátký zápis s ukázkami výstupů a poznatky z tréninku.

## Technické prostředí

Framework je PyTorch s backendem `mps` (GPU Apple Silicon), Python 3.11+. Alternativa je MLX, pokud chceš experimentovat s nativním frameworkem od Applu.

| Položka | Volba | Poznámka |
| --- | --- | --- |
| Hardware | MacBook Air M2 | Bez větráku, při delším tréninku se přiškrcuje. Pouštěj na podložce, ideálně v noci. |
| Framework | PyTorch (`device="mps"`) | Přes `torch.backends.mps.is_available()` ověř GPU. |
| Knihovny | torch, torchvision, numpy, pillow, tqdm, matplotlib | Žádný diffusers ani hotové modely. |
| Logování | TensorBoard nebo obyčejné PNG mřížky | Každých N kroků ulož vzorek, ať vidíš pokrok. |
| Velikost modelu | 5–20 M parametrů | Na 8 GB RAM drž batch 64–128. |

## Dataset

Potřebuješ aspoň 5 000 spritů, ideálně 20 000 a víc. Hledej volně licencované sady (CC0 nebo CC-BY) a licenci si u každé ověř.

Zdroje k prozkoumání:

- Kaggle: datasety pixel-art postav a spritů (hledej „pixel art characters“).
- OpenGameArt.org: sprity s licencí CC0, často jako spritesheety k rozřezání.
- Vlastní generátor: procedurálně skládané postavičky z částí (hlava, tělo, barvy). Je to dobrá záloha a je to zábava navíc.

Předzpracování:

1. Spritesheety rozřež na jednotlivé dlaždice a zahoď prázdné.
2. Sjednoť velikost na 32×32 metodou nearest neighbor, nikdy ne bilineárně, jinak se pixely rozmažou.
3. Průhlednost nahraď jednotnou barvou pozadí, nebo ponech 4 kanály RGBA.
4. Hodnoty normalizuj do rozsahu \[-1, 1\] a ulož vše jako jeden tensor `.pt` kvůli rychlému načítání.
5. Augmentace: jen horizontální zrcadlení. Rotace a škálování pixel art ničí.

## Architektura modelu

Model je DDPM (Denoising Diffusion Probabilistic Model) s malým U-Netem. Učí se odhadovat šum přidaný k obrázku; generování pak postupně odšumuje čistý náhodný šum.

Dopředný proces přidává šum podle rozvrhu β v T krocích:

```latex
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

Trénovací ztráta je MSE mezi skutečným a odhadnutým šumem:

```latex
L = \mathbb{E}_{x_0, t, \epsilon} \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2
```

| Komponenta | Doporučení |
| --- | --- |
| Počet kroků T | 1 000 při tréninku; při generování DDIM s 50 kroky |
| Rozvrh šumu | Kosinový (lépe funguje pro malé obrázky než lineární) |
| U-Net | 3 úrovně: 32→16→8 px, kanály 64/128/256 |
| Bloky | ResNet bloky s GroupNorm a SiLU, self-attention jen na 8×8 |
| Embedding času | Sinusové kódování t, přidané do každého ResNet bloku |
| Optimalizátor | AdamW, lr 2e-4, EMA vah s decay 0,999 |

Moduly k napsání: `data.py`, `unet.py`, `diffusion.py` (rozvrh, forward, sampling), `train.py`, `sample.py`.

## Fáze a milníky

Celý projekt je zhruba na 4–6 večerů plus noční tréninky. Každá fáze končí něčím, co uvidíš.

- [ ] **1. Data (1 večer):** stáhnout, rozřezat a uložit dataset. Milník: mřížka 64 náhodných spritů z datasetu.
- [ ] **2. Dopředný proces (krátký večer):** rozvrh šumu a funkce `q_sample`. Milník: obrázek, jak sprite postupně mizí v šumu pro t = 0…1000.
- [ ] **3. U-Net (1 večer):** model a kontrola tvarů tensorů. Milník: forward pass na batchi projde, počet parametrů je 5–20 M.
- [ ] **4. Trénink (1 večer + noc):** trénovací smyčka s EMA a checkpointy. Milník: loss klesá a ukládá se mřížka vzorků každých 2 000 kroků.
- [ ] **5. Sampling (1 večer):** DDPM a pak DDIM se 50 kroky. Milník: rozpoznatelné postavičky.
- [ ] **6. Ladění a zápis:** porovnat výsledky a sepsat poznatky.

Tip: nejdřív model přetrénuj na 64 obrázcích. Když nedokáže zreprodukovat ani je, chyba je v kódu, ne v datech.

## Kritéria úspěchu a rozšíření

Projekt je hotový, když ve vygenerované mřížce 64 spritů je většina rozpoznatelných postaviček s ostrými pixely a nejde o kopie z datasetu.

- Kopie ověř hledáním nejbližšího souseda v datasetu (L2 vzdálenost) pro pár vzorků.
- Ostrost: výstup kvantuj na paletu z datasetu (např. 16–32 barev k-means), ať nejsou rozmazané přechody.

Rozšíření, až základ funguje:

- **Podmíněné generování:** třída (rytíř, mág, zvíře) jako vstup a classifier-free guidance.
- **Interpolace:** plynulý přechod mezi dvěma sprity přes latentní šum (DDIM je deterministický).
- **Inpainting:** dokreslení části spritu, třeba vyměnit hlavu.
- **Webové demo:** export do ONNX nebo Core ML a generování přímo v prohlížeči či aplikaci.
- **64×64 nebo animace:** generování krátkých animačních cyklů (4 snímky chůze).
