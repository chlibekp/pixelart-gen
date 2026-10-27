"""Příprava datasetu 32×32 spritů.

Dva zdroje dat:
  1. Procedurální generátor postaviček (rytíř, mág, zloděj, vesničan, sliz, zvíře).
     Vše je kreslené kódem, takže licence není problém.
  2. Volitelně vlastní spritesheety (PNG) ze složky --sheets, rozřezané na dlaždice.

Výstup: data/sprites.pt  {"images": uint8 [N,3,32,32], "labels": int64 [N], "classes": [...], "bg": (r,g,b)}
         data/dataset_grid.png  (mřížka 64 náhodných spritů, milník fáze 1)

Použití:
  python data.py --n 20000
  python data.py --n 20000 --sheets data/raw --tile 16
"""
import argparse
import colorsys
import glob
import os

import numpy as np
import torch
from PIL import Image

SIZE = 32
BG = (226, 222, 212)  # jednotné pozadí místo průhlednosti
OUTLINE = (24, 20, 30)
CLASSES = ["knight", "mage", "rogue", "villager", "slime", "beast"]
EXTERNAL_LABEL = len(CLASSES)  # sprity ze spritesheetů nemají třídu

SKIN = [(255, 219, 172), (241, 194, 125), (224, 172, 105), (198, 134, 66),
        (141, 85, 36), (120, 180, 100), (180, 190, 220), (220, 220, 210)]
HAIR = [(30, 25, 25), (90, 55, 30), (150, 100, 50), (230, 200, 110),
        (190, 70, 30), (235, 235, 235), (120, 120, 130)]
METAL = [(170, 175, 185), (140, 145, 160), (200, 170, 80), (120, 90, 60), (90, 95, 110)]
WOOD = [(120, 80, 40), (95, 60, 30), (150, 105, 60)]


# ---------------------------------------------------------------- barvy

def rand_color(rng, s=(0.45, 0.9), v=(0.45, 0.95), h=None):
    h = rng.random() if h is None else h
    r, g, b = colorsys.hsv_to_rgb(h, rng.uniform(*s), rng.uniform(*v))
    return (int(r * 255), int(g * 255), int(b * 255))


def dark(c, f=0.68):
    return tuple(int(x * f) for x in c)


def light(c):
    return tuple(min(255, int(x * 1.15 + 28)) for x in c)


def pick(rng, items):
    return items[rng.integers(len(items))]


# ---------------------------------------------------------------- plátno

class Canvas:
    """Kreslí na větší plátno (48×48); na konci se postava ořízne a vycentruje do 32×32."""

    def __init__(self, size=48):
        self.size = size
        self.rgb = np.zeros((size, size, 3), np.uint8)
        self.mask = np.zeros((size, size), bool)

    def px(self, x, y, c):
        if 0 <= x < self.size and 0 <= y < self.size:
            self.rgb[y, x] = c
            self.mask[y, x] = True

    def rect(self, x0, y0, x1, y1, c, shade=True):
        """Vyplněný obdélník (krajní body včetně); levý sloupec světlejší, pravý tmavší."""
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                col = c
                if shade and x1 - x0 >= 2:
                    if x == x0:
                        col = light(c)
                    elif x == x1:
                        col = dark(c)
                self.px(x, y, col)

    def ellipse(self, cx, cy, rx, ry, c):
        """Elipsa se stínováním: vlevo nahoře světlo, vpravo dole stín."""
        for y in range(int(cy - ry) - 1, int(cy + ry) + 2):
            for x in range(int(cx - rx) - 1, int(cx + rx) + 2):
                dx, dy = (x + 0.5 - cx) / rx, (y + 0.5 - cy) / ry
                d = dx * dx + dy * dy
                if d <= 1:
                    col = c
                    if dx + dy > 0.7:
                        col = dark(c)
                    elif dx + dy < -0.9 and d > 0.35:
                        col = light(c)
                    self.px(x, y, col)

    def outline(self, c=OUTLINE):
        m = self.mask
        p = np.pad(m, 1)
        edge = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~m
        self.rgb[edge] = c
        self.mask = m | edge

    def to_sprite(self):
        """Ořízne bbox postavy a umístí ji do 32×32: vodorovně na střed, spodkem na řádek 30."""
        ys, xs = np.nonzero(self.mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0 + 1, x1 - x0 + 1
        if h > SIZE - 1 or w > SIZE - 1:
            return None
        out = np.empty((SIZE, SIZE, 3), np.uint8)
        out[:] = BG
        oy = SIZE - 1 - h
        ox = (SIZE - w) // 2
        crop_rgb = self.rgb[y0:y1 + 1, x0:x1 + 1]
        crop_m = self.mask[y0:y1 + 1, x0:x1 + 1]
        region = out[oy:oy + h, ox:ox + w]
        region[crop_m] = crop_rgb[crop_m]
        return out


# ---------------------------------------------------------------- humanoidi

def draw_humanoid(rng, kind):
    c = Canvas()
    cx = 24  # osa symetrie leží mezi pixely 23 a 24
    hw = int(rng.integers(3, 5))         # polovina šířky hlavy
    tw = int(rng.integers(4, 6))         # polovina šířky trupu
    head_h = int(rng.integers(6, 8))
    torso_h = int(rng.integers(7, 10))
    leg_h = int(rng.integers(5, 8))
    head_top = 14
    head_bot = head_top + head_h - 1
    torso_top = head_bot + 1
    torso_bot = torso_top + torso_h - 1
    leg_bot = torso_bot + leg_h
    eye_y = head_top + head_h // 2

    L = lambda half: cx - half          # levý okraj prvku s danou poloviční šířkou
    R = lambda half: cx - 1 + half      # pravý okraj

    skin = pick(rng, SKIN)
    if kind == "knight":
        shirt = pick(rng, METAL) if rng.random() < 0.7 else rand_color(rng)
        pants = pick(rng, METAL)
    elif kind == "rogue":
        shirt = rand_color(rng, s=(0.3, 0.7), v=(0.25, 0.55))
        pants = rand_color(rng, s=(0.2, 0.5), v=(0.2, 0.45))
    else:
        shirt = rand_color(rng)
        pants = rand_color(rng, s=(0.2, 0.7), v=(0.3, 0.7))
    boots = dark(pick(rng, WOOD), 0.8)
    robe = kind == "mage" and rng.random() < 0.8

    # plášť (za tělem)
    if rng.random() < {"knight": 0.45, "mage": 0.3, "rogue": 0.5, "villager": 0.1}[kind]:
        cape = rand_color(rng, v=(0.35, 0.75))
        c.rect(L(tw + 3), torso_top, R(tw + 3), min(leg_bot - 1, torso_bot + 4), dark(cape, 0.85))

    # nohy / róba
    if robe:
        for i, y in enumerate(range(torso_top, leg_bot + 1)):
            widen = min(2, i // 5)
            c.rect(L(tw + widen), y, R(tw + widen), y, shirt)
        c.rect(L(2), leg_bot, L(1), leg_bot, boots, shade=False)
        c.rect(R(1), leg_bot, R(2), leg_bot, boots, shade=False)
    else:
        c.rect(L(tw - 1), torso_bot + 1, cx - 2, leg_bot, pants)
        c.rect(cx + 1, torso_bot + 1, R(tw - 1), leg_bot, pants)
        bh = int(rng.integers(1, 3))
        c.rect(L(tw - 1), leg_bot - bh + 1, cx - 2, leg_bot, boots, shade=False)
        c.rect(cx + 1, leg_bot - bh + 1, R(tw - 1), leg_bot, boots, shade=False)

    # trup
    c.rect(L(tw), torso_top, R(tw), torso_bot, shirt)
    if kind == "knight" and rng.random() < 0.6:   # náprsní plát
        c.rect(L(tw - 2), torso_top + 1, R(tw - 2), torso_bot - 2, light(shirt), shade=False)
    if rng.random() < 0.6:                        # opasek
        belt_y = torso_bot - 1 if not robe else torso_top + torso_h // 2
        c.rect(L(tw), belt_y, R(tw), belt_y, dark(pick(rng, WOOD)), shade=False)
        c.px(cx - 1, belt_y, (230, 200, 80))
        c.px(cx, belt_y, (230, 200, 80))
    if kind == "villager" and rng.random() < 0.4:  # zástěra
        c.rect(L(tw - 2), torso_top + 3, R(tw - 2), torso_bot + 2, (235, 230, 215), shade=False)

    # ruce
    sleeve = shirt if rng.random() < 0.75 else skin
    arm_l0, arm_l1 = L(tw + 2), L(tw + 1)
    arm_r0, arm_r1 = R(tw + 1), R(tw + 2)
    hand_y = torso_bot - 1
    c.rect(arm_l0, torso_top + 1, arm_l1, hand_y - 1, dark(sleeve, 0.9), shade=False)
    c.rect(arm_r0, torso_top + 1, arm_r1, hand_y - 1, dark(sleeve, 0.8), shade=False)
    c.rect(arm_l0, hand_y, arm_l1, hand_y, skin, shade=False)
    c.rect(arm_r0, hand_y, arm_r1, hand_y, skin, shade=False)

    # hlava a oči
    c.rect(L(hw), head_top, R(hw), head_bot, skin)
    eye = (20, 20, 30) if rng.random() < 0.85 else rand_color(rng, s=(0.7, 1), v=(0.8, 1))
    c.px(cx - 2, eye_y, eye)
    c.px(cx + 1, eye_y, eye)
    if rng.random() < 0.4:
        c.px(cx - 1, head_bot - 1, dark(skin, 0.6))
        c.px(cx, head_bot - 1, dark(skin, 0.6))

    # vous
    if rng.random() < {"mage": 0.45, "villager": 0.2}.get(kind, 0.1):
        beard = (235, 235, 235) if kind == "mage" and rng.random() < 0.6 else pick(rng, HAIR)
        c.rect(L(hw), head_bot - 1, R(hw), head_bot, beard, shade=False)
        c.rect(L(hw - 1), head_bot + 1, R(hw - 1), head_bot + 2 + int(rng.integers(0, 3)), beard)

    # pokrývka hlavy nebo vlasy
    hats = {"knight": ["helmet"] * 6 + ["hair"] * 3 + ["crown"],
            "mage": ["wizard"] * 6 + ["hood"] * 3 + ["hair"],
            "rogue": ["hood"] * 6 + ["hair"] * 3 + ["bandana"],
            "villager": ["hair"] * 7 + ["straw"] * 2 + ["bandana"]}[kind]
    hat = pick(rng, hats)
    hair = pick(rng, HAIR) if rng.random() < 0.85 else rand_color(rng)

    if hat == "hair":
        style = int(rng.integers(4))
        if style == 0:      # krátké
            c.rect(L(hw), head_top - 1, R(hw), head_top, hair)
            c.px(L(hw), head_top + 1, hair)
            c.px(R(hw), head_top + 1, hair)
        elif style == 1:    # dlouhé
            c.rect(L(hw), head_top - 1, R(hw), head_top, hair)
            c.rect(L(hw + 1), head_top, L(hw), head_bot + 2, hair, shade=False)
            c.rect(R(hw), head_top, R(hw + 1), head_bot + 2, dark(hair, 0.85), shade=False)
        elif style == 2:    # bodlinky
            c.rect(L(hw), head_top - 1, R(hw), head_top, hair)
            for x in range(L(hw), R(hw) + 1, 2):
                c.px(x, head_top - 2, hair)
        else:               # drdol
            c.rect(L(hw), head_top - 1, R(hw), head_top, hair)
            c.rect(cx - 1, head_top - 3, cx, head_top - 2, hair, shade=False)
    elif hat == "helmet":
        metal = pick(rng, METAL)
        c.rect(L(hw + 1), head_top - 1, R(hw + 1), head_bot - 1, metal)
        c.rect(L(hw), eye_y, R(hw), eye_y, OUTLINE, shade=False)  # průzor
        if rng.random() < 0.5:  # chochol
            plume = rand_color(rng, s=(0.7, 1), v=(0.6, 0.95))
            c.rect(cx - 1, head_top - 4, cx, head_top - 2, plume, shade=False)
        if rng.random() < 0.4:  # rohy
            c.px(L(hw + 2), head_top - 2, (235, 230, 210))
            c.px(R(hw + 2), head_top - 2, (235, 230, 210))
            c.px(L(hw + 2), head_top - 1, (235, 230, 210))
            c.px(R(hw + 2), head_top - 1, (235, 230, 210))
    elif hat == "wizard":
        col = rand_color(rng, v=(0.35, 0.8))
        c.rect(L(hw + 2), head_top - 1, R(hw + 2), head_top - 1, col, shade=False)  # krempa
        cone_h = int(rng.integers(5, 8))
        bend = int(rng.integers(-1, 2))
        for i in range(cone_h):
            half = max(1, hw - (i * hw) // cone_h)
            shift = bend if i > cone_h * 0.6 else 0
            c.rect(L(half) + shift, head_top - 2 - i, R(half) + shift, head_top - 2 - i, col)
        if rng.random() < 0.5:  # hvězdička
            c.px(cx - 1, head_top - 3, (250, 230, 90))
    elif hat == "hood":
        col = rand_color(rng, s=(0.2, 0.7), v=(0.25, 0.65))
        c.rect(L(hw + 1), head_top - 2, R(hw + 1), head_bot + 1, col)
        c.rect(L(hw - 1), eye_y - 1, R(hw - 1), head_bot, skin, shade=False)
        c.px(cx - 2, eye_y, eye)
        c.px(cx + 1, eye_y, eye)
        c.rect(L(hw - 1), eye_y - 1, R(hw - 1), eye_y - 1, dark(col, 0.5), shade=False)
    elif hat == "crown":
        gold = (235, 195, 60)
        c.rect(L(hw), head_top - 1, R(hw), head_top, gold, shade=False)
        for x in range(L(hw), R(hw) + 1, 2):
            c.px(x, head_top - 2, gold)
        c.px(cx - 1, head_top, (200, 40, 60))
    elif hat == "bandana":
        col = rand_color(rng, s=(0.6, 0.9))
        c.rect(L(hw), head_top - 1, R(hw), head_top + 1, col)
        c.px(R(hw + 1), head_top + 2, col)
        c.px(R(hw + 2), head_top + 3, col)
    elif hat == "straw":
        straw = (225, 190, 100)
        c.rect(L(hw + 3), head_top, R(hw + 3), head_top, straw, shade=False)
        c.rect(L(hw), head_top - 2, R(hw), head_top - 1, straw)

    # předměty v rukou
    items = {"knight": ["sword"] * 5 + ["axe"] * 2 + ["none"],
             "mage": ["staff"] * 7 + ["book"] * 2 + ["none"],
             "rogue": ["dagger"] * 4 + ["bow"] * 4 + ["none"],
             "villager": ["none"] * 4 + ["fork"] * 2 + ["lantern"] * 2}[kind]
    item = pick(rng, items)
    hx = arm_r1 + 1
    if item == "sword":
        blade = (215, 220, 230)
        c.rect(hx, hand_y - 9, hx, hand_y - 2, blade, shade=False)
        c.rect(hx - 1, hand_y - 1, hx + 1, hand_y - 1, pick(rng, METAL[2:4]), shade=False)
        c.px(hx, hand_y, pick(rng, WOOD))
        c.px(hx, hand_y + 1, (200, 170, 80))
    elif item == "axe":
        c.rect(hx, hand_y - 8, hx, hand_y + 2, pick(rng, WOOD), shade=False)
        c.rect(hx + 1, hand_y - 8, hx + 3, hand_y - 5, (180, 185, 195))
    elif item == "dagger":
        c.rect(hx, hand_y - 4, hx, hand_y - 1, (215, 220, 230), shade=False)
        c.px(hx, hand_y, pick(rng, WOOD))
    elif item == "staff":
        c.rect(hx, head_top - 3, hx, leg_bot, pick(rng, WOOD), shade=False)
        gem = rand_color(rng, s=(0.7, 1), v=(0.85, 1))
        c.rect(hx - 1, head_top - 5, hx + 1, head_top - 3, gem)
    elif item == "book":
        bc = rand_color(rng, v=(0.3, 0.7))
        c.rect(hx - 1, hand_y - 3, hx + 2, hand_y, bc)
        c.rect(hx, hand_y - 2, hx + 1, hand_y - 1, (240, 235, 210), shade=False)
    elif item == "bow":
        wood = pick(rng, WOOD)
        for i, dy in enumerate(range(-6, 7)):
            dx = 2 - (abs(dy) * abs(dy)) // 14
            c.px(hx + dx, hand_y + dy - 2, wood)
        c.rect(hx, hand_y - 8, hx, hand_y + 4, (230, 230, 220), shade=False)
    elif item == "fork":
        c.rect(hx, hand_y - 9, hx, leg_bot, pick(rng, WOOD), shade=False)
        c.rect(hx - 1, hand_y - 10, hx + 1, hand_y - 10, (170, 175, 185), shade=False)
        for x in (hx - 1, hx + 1):
            c.rect(x, hand_y - 12, x, hand_y - 11, (170, 175, 185), shade=False)
    elif item == "lantern":
        c.px(hx, hand_y + 1, (60, 60, 70))
        c.rect(hx - 1, hand_y + 2, hx + 1, hand_y + 4, (250, 210, 90))

    # štít v levé ruce
    if kind == "knight" and rng.random() < 0.55:
        sc = rand_color(rng)
        sx1 = arm_l0 + 1
        c.rect(sx1 - 4, torso_top + 1, sx1, torso_top + 6, sc)
        c.rect(sx1 - 3, torso_top + 7, sx1 - 1, torso_top + 7, sc, shade=False)
        emb = rand_color(rng, s=(0.1, 0.4), v=(0.9, 1))
        c.rect(sx1 - 2, torso_top + 2, sx1 - 2, torso_top + 6, emb, shade=False)
        c.rect(sx1 - 3, torso_top + 3, sx1 - 1, torso_top + 3, emb, shade=False)

    c.outline()
    return c


# ---------------------------------------------------------------- tvorové

def draw_slime(rng):
    c = Canvas()
    col = rand_color(rng, s=(0.5, 0.95), v=(0.55, 0.95))
    rx = rng.uniform(5.5, 10)
    ry = rng.uniform(4.5, min(8, rx))
    cx, base = 24.0, 36
    cy = base - ry
    c.ellipse(cx, cy, rx, ry, col)
    c.rect(int(cx - rx) + 1, base - 1, int(cx + rx) - 1, base, dark(col, 0.75), shade=False)  # plochý spodek
    if rng.random() < 0.5:  # špička nahoře
        c.rect(int(cx) - 1, int(cy - ry) - 1, int(cx), int(cy - ry), col, shade=False)
        c.px(int(cx), int(cy - ry) - 2, col)
    hl = light(light(col))
    c.px(int(cx - rx * 0.5), int(cy - ry * 0.5), hl)
    c.px(int(cx - rx * 0.5) + 1, int(cy - ry * 0.5), hl)
    ey = int(cy) + int(rng.integers(-1, 2))
    gap = int(rng.integers(2, 4))
    for ex in (int(cx) - gap - 1, int(cx) + gap):
        c.px(ex, ey, (20, 20, 30))
        c.px(ex, ey - 1, (20, 20, 30))
        c.px(ex + 1 if ex < cx else ex - 1, ey - 1, (250, 250, 250)) if rng.random() < 0.5 else None
    if rng.random() < 0.6:  # pusa
        c.rect(int(cx) - 1, ey + 2, int(cx), ey + 2, dark(col, 0.4), shade=False)
    if rng.random() < 0.15:  # korunka
        top = int(cy - ry) - 1
        c.rect(int(cx) - 2, top - 1, int(cx) + 1, top, (235, 195, 60), shade=False)
        c.px(int(cx) - 2, top - 2, (235, 195, 60))
        c.px(int(cx) + 1, top - 2, (235, 195, 60))
    c.outline()
    return c


def draw_beast(rng):
    """Čtyřnohé zvíře z boku, kouká doprava (zrcadlení v augmentaci obstará opačný směr)."""
    c = Canvas()
    fur = rand_color(rng, s=(0.2, 0.7), v=(0.35, 0.9), h=rng.uniform(0.0, 0.15) if rng.random() < 0.7 else None)
    bw = int(rng.integers(6, 9))        # polovina délky těla
    bh = int(rng.integers(3, 5))        # polovina výšky těla
    leg_h = int(rng.integers(3, 6))
    base = 36
    body_cy = base - leg_h - bh
    cx = 22
    # nohy (zadní tmavší)
    for lx, f in ((cx - bw + 1, 0.7), (cx - bw + 3, 0.85), (cx + bw - 4, 0.7), (cx + bw - 2, 0.85)):
        c.rect(lx, body_cy + bh - 1, lx + 1, base, dark(fur, f), shade=False)
    # ocas
    tail = int(rng.integers(3))
    tx = cx - bw - 1
    if tail == 0:
        for i in range(4):
            c.px(tx - i // 2, body_cy - 1 - i, fur)
    elif tail == 1:
        c.rect(tx - 2, body_cy - 1, tx, body_cy, fur, shade=False)
    else:
        c.rect(tx - 1, body_cy - 2, tx, body_cy + 1, light(fur), shade=False)
    # tělo
    c.ellipse(cx, body_cy, bw, bh, fur)
    if rng.random() < 0.4:  # skvrny / pruhy
        spot = dark(fur, 0.6) if rng.random() < 0.5 else light(fur)
        for _ in range(int(rng.integers(2, 5))):
            c.px(int(rng.integers(cx - bw + 2, cx + bw - 2)), int(rng.integers(body_cy - bh + 1, body_cy + 1)), spot)
    if rng.random() < 0.5:  # světlé břicho
        c.rect(cx - bw + 3, body_cy + bh - 1, cx + bw - 3, body_cy + bh - 1, light(fur), shade=False)
    # hlava
    hs = int(rng.integers(3, 5))
    hx0 = cx + bw - 2
    hy1 = body_cy - bh + 3
    c.rect(hx0, hy1 - hs * 2 + 1, hx0 + hs * 2 - 1, hy1, fur)
    snout = int(rng.integers(1, 4))
    c.rect(hx0 + hs * 2, hy1 - hs + 1, hx0 + hs * 2 + snout - 1, hy1, light(fur), shade=False)
    c.px(hx0 + hs * 2 + snout - 1, hy1 - hs + 1, (30, 25, 25))  # čumák
    c.px(hx0 + hs * 2 - 2, hy1 - hs * 2 + 3, (20, 20, 30))      # oko
    # uši
    ear = int(rng.integers(3))
    ex = hx0 + 1
    top = hy1 - hs * 2
    if ear == 0:    # špičaté
        c.px(ex, top, dark(fur, 0.85))
        c.px(ex + 1, top, dark(fur, 0.85))
        c.px(ex, top - 1, dark(fur, 0.85))
    elif ear == 1:  # dlouhé (zajíc)
        c.rect(ex, top - 4, ex + 1, top, dark(fur, 0.9), shade=False)
    else:           # parohy
        antler = (200, 180, 140)
        c.rect(ex + 1, top - 3, ex + 1, top, antler, shade=False)
        c.px(ex, top - 2, antler)
        c.px(ex + 2, top - 3, antler)
    c.outline()
    return c


def generate_sprite(rng, label):
    kind = CLASSES[label]
    if kind == "slime":
        return draw_slime(rng)
    if kind == "beast":
        return draw_beast(rng)
    return draw_humanoid(rng, kind)


def generate(n, seed=0):
    rng = np.random.default_rng(seed)
    weights = np.array([0.22, 0.22, 0.18, 0.18, 0.1, 0.1])
    images, labels = [], []
    while len(images) < n:
        label = int(rng.choice(len(CLASSES), p=weights))
        sprite = generate_sprite(rng, label).to_sprite()
        if sprite is None:  # nevešlo se do 32×32
            continue
        images.append(sprite)
        labels.append(label)
    return np.stack(images), np.array(labels)


# ---------------------------------------------------------------- spritesheety

def slice_sheets(folder, tile):
    """Rozřeže všechny PNG ve složce na dlaždice tile×tile, zahodí prázdné, zmenší/zvětší na 32×32 metodou nearest."""
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)):
        sheet = np.array(Image.open(path).convert("RGBA"))
        H, W = sheet.shape[:2]
        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                t = sheet[y:y + tile, x:x + tile]
                alpha = t[..., 3] > 127
                if alpha.mean() < 0.05 or alpha.all() and t[..., :3].std() < 4:
                    continue
                t = np.array(Image.fromarray(t).resize((SIZE, SIZE), Image.NEAREST))
                rgb = np.empty((SIZE, SIZE, 3), np.uint8)
                rgb[:] = BG
                m = t[..., 3] > 127
                rgb[m] = t[..., :3][m]
                out.append(rgb)
    return np.stack(out) if out else np.zeros((0, SIZE, SIZE, 3), np.uint8)


# ---------------------------------------------------------------- načítání a vizualizace

def load_dataset(path="data/sprites.pt", device="cpu"):
    """Vrátí float tensor [N,3,32,32] v rozsahu [-1, 1] a labely."""
    d = torch.load(path)
    x = d["images"].to(device).float() / 127.5 - 1
    return x, d["labels"].to(device)


def random_flip(x):
    """Jediná augmentace: horizontální zrcadlení poloviny batche."""
    flip = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < 0.5
    return torch.where(flip, x.flip(3), x)


def to_uint8(x):
    """[-1,1] float [N,3,H,W] -> uint8 [N,H,W,3]."""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()


def save_grid(x, path, nrow=8, scale=4, pad=1):
    """Uloží mřížku obrázků; zvětšení nearest neighbor, aby pixely zůstaly ostré."""
    imgs = to_uint8(x) if torch.is_tensor(x) else x
    n, h, w, _ = imgs.shape
    ncol = (n + nrow - 1) // nrow
    grid = np.full((ncol * (h + pad) + pad, nrow * (w + pad) + pad, 3), 60, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, nrow)
        grid[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = im
    img = Image.fromarray(grid)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000, help="počet procedurálních spritů")
    p.add_argument("--sheets", default=None, help="složka se spritesheety (PNG)")
    p.add_argument("--tile", type=int, default=16, help="velikost dlaždice ve spritesheetech")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/sprites.pt")
    args = p.parse_args()

    images, labels = generate(args.n, args.seed)
    if args.sheets:
        ext = slice_sheets(args.sheets, args.tile)
        print(f"spritesheety: {len(ext)} dlaždic")
        images = np.concatenate([images, ext])
        labels = np.concatenate([labels, np.full(len(ext), EXTERNAL_LABEL)])

    uniq = len(np.unique(images.reshape(len(images), -1), axis=0))
    print(f"spritů: {len(images)}, unikátních: {uniq}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"images": torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
                "labels": torch.from_numpy(labels).long(),
                "classes": CLASSES, "bg": BG}, args.out)
    idx = np.random.default_rng(1).choice(len(images), 64, replace=False)
    save_grid(images[idx], os.path.join(os.path.dirname(args.out), "dataset_grid.png"))
    print(f"uloženo do {args.out}")


if __name__ == "__main__":
    main()
