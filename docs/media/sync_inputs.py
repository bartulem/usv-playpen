"""
@author: bartulem

Builds the Process-page figure of the synchronization signals: which input of each
recording device receives the Loopbio Triggerbox camera-frame pulses and which
receives the Arduino SYNC pulses, with the ~2.3 s Triggerbox pause that marks the
tracking start, the tracking end N pulses later, and a true-scale zoom of the
Triggerbox pulse train.

The figure is an inline SVG (not a PNG) so that its text inherits the page font
(Lato in sphinx_rtd_theme) and its colors follow the sphinx_rtd_dark_mode switch
(``html[data-theme="dark"]``). ``docs/Process.rst`` embeds the written fragment with
``.. raw:: html`` and ``:file: media/sync_inputs.html``.

The pulse timings drawn at true scale were measured on session 20260421_182605:
the Triggerbox pulse is 0.116 ms high every 6.66 ms (ch04 lowest bit, 250 kHz) and
the Arduino pulse is 250 ms high with 252-1493 ms gaps (ch02 lowest bit). The main
traces are schematic: the time axis is broken and the Triggerbox ticks there are
drawn sparser than 150/s.

Usage
-----
python docs/media/sync_inputs.py docs/media [--demo]

Writes
------
<out_dir>/sync_inputs.html
    The fragment to embed (a <style> block plus the <svg>). The prettier pre-commit
    hook reformats it on commit; the reformatted file renders identically.
<out_dir>/sync_inputs_demo.html
    Only with ``--demo``: a preview page showing the fragment on the theme's light
    and dark backgrounds.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

out_dir = pathlib.Path(sys.argv[1])

# time layout (s); the recording is cut by an axis break
pre_end = 3.0
pause = 2.3
t_start = pre_end + pause
part1 = 6.0
break_at = t_start + part1
gap_vis = 0.45

# Arduino pulses: 250 ms high, random 250-1500 ms gaps; the sequence starts at 0.5 s
# so that the tracking start does not land on a pulse
rng = np.random.default_rng(7)
ar_all = []
t = 0.5
while t < 20.0:
    if not (break_at - 0.25 < t < break_at + gap_vis):
        ar_all.append(t)
    t += 0.25 + rng.uniform(0.25, 1.5)

# the tracking end sits in the middle of the Arduino gap nearest 1.5 s after the axis
# break, so it never coincides with an LED event
target_end = break_at + gap_vis + 1.5
gaps = [(a + 0.25, b) for a, b in itertools.pairwise(ar_all) if a > break_at + gap_vis]
gap_lo, gap_hi = min(gaps, key=lambda g: abs((g[0] + g[1]) / 2 - target_end))
t_end = (gap_lo + gap_hi) / 2
t_max = t_end + 1.2
ar_on = [tt for tt in ar_all if tt + 0.25 <= t_max]
if any(tt - 0.1 < t_start < tt + 0.35 for tt in ar_on):
    msg = 'the tracking start lands on an Arduino pulse; change the sequence start'
    raise ValueError(msg)

# geometry (px)
W = 960
X_LABEL = 0
X0 = 400
X1 = W - 10
PX_PER_S = (X1 - X0) / t_max
ROW_PITCH = 64
TRACE_H = 24
Y_FIRST = 168


def x_of(t):
    """Converts seconds on the drawn axis to an x pixel position."""
    return X0 + t * PX_PER_S


rows = [
    ('USGH M and S · digital input 4', 'Triggerbox; stored in the lowest bit of ch04', 'tb'),
    ('USGH M and S · digital input 2', 'Arduino SYNC; stored in the lowest bit of ch02', 'ar'),
    ('Neuropixels imec0/imec1 · SY bit', 'Triggerbox; the probe sync channel', 'tb'),
    ('NIDQ · bit 5', 'Triggerbox (optional)', 'tb'),
    ('NIDQ · bit 7', 'Arduino SYNC (optional)', 'ar'),
    ('Cameras · IR sync LED', 'Arduino SYNC; LED brightness in each frame', 'led'),
]

tick_step = 0.09
tb_times = np.concatenate([np.arange(0.05, pre_end, tick_step),
                           np.arange(t_start, break_at, tick_step),
                           np.arange(break_at + gap_vis, t_max, tick_step)])

parts = []
add = parts.append


def f(v):
    """Formats a coordinate with one decimal."""
    return f'{v:.1f}'


for i, (label, sub, kind) in enumerate(rows):
    y_mid = Y_FIRST + i * ROW_PITCH
    base = y_mid + TRACE_H / 2
    top = base - TRACE_H
    add(f'<text class="sf-label" x="{X_LABEL}" y="{f(y_mid - 4)}">{label}</text>')
    add(f'<text class="sf-sub" x="{X_LABEL}" y="{f(y_mid + 17)}">{sub}</text>')
    cls = 'sf-tb' if kind == 'tb' else 'sf-ar'
    if kind == 'led':
        add(f'<line class="sf-ar sf-base" x1="{f(x_of(t_start))}" y1="{f(base)}" x2="{f(x_of(t_end))}" y2="{f(base)}"/>')
        for a, b in ((0, t_start), (t_end, t_max)):
            add(f'<line class="sf-none" x1="{f(x_of(a))}" y1="{f(base)}" x2="{f(x_of(b))}" y2="{f(base)}"/>')
        add(f'<text class="sf-note" x="{f(x_of(t_start / 2))}" y="{f(base - 6)}" text-anchor="middle">no video</text>')
    else:
        add(f'<line class="{cls} sf-base" x1="{f(x_of(0))}" y1="{f(base)}" x2="{f(x_of(t_max))}" y2="{f(base)}"/>')
    if kind == 'tb':
        d = ''.join(f'M{f(x_of(tt))} {f(base)}V{f(top)}' for tt in tb_times)
        add(f'<path class="sf-tb sf-tick" d="{d}"/>')
    else:
        d = ''
        for tt in ar_on:
            xa, xb = x_of(tt), x_of(tt + 0.25)
            if kind == 'led':
                if not (t_start <= tt and tt + 0.25 <= t_end):
                    continue
                xm = x_of(tt + 0.03)
                ym = base - TRACE_H * 0.55
                d += f'M{f(xa)} {f(base)}V{f(ym)}H{f(xm)}V{f(top)}H{f(xb)}V{f(base)}'
            else:
                d += f'M{f(xa)} {f(base)}V{f(top)}H{f(xb)}V{f(base)}'
        add(f'<path class="sf-ar sf-step{" sf-led" if kind == "led" else ""}" d="{d}"/>')
    # axis break
    xb0 = x_of(break_at)
    add(f'<rect class="sf-bg" x="{f(xb0 + 2)}" y="{f(top - 4)}" width="{f(gap_vis * PX_PER_S - 4)}" height="{f(TRACE_H + 8)}"/>')
    for dx in (5, 13):
        add(f'<line class="sf-break" x1="{f(xb0 + dx)}" y1="{f(base + 4)}" x2="{f(xb0 + dx + 6)}" y2="{f(top - 4)}"/>')

last_base = Y_FIRST + (len(rows) - 1) * ROW_PITCH + TRACE_H / 2
first_top = Y_FIRST - TRACE_H / 2

# ~2.3 s pause bracket above the first row
yb = first_top - 10
add(f'<line class="sf-tb sf-bracket" x1="{f(x_of(pre_end))}" y1="{f(yb)}" x2="{f(x_of(t_start))}" y2="{f(yb)}" '
    'marker-start="url(#sf-arr-tb)" marker-end="url(#sf-arr-tb)"/>')
add(f'<text class="sf-note sf-tbtext" x="{f(x_of((pre_end + t_start) / 2))}" y="{f(yb - 7)}" text-anchor="middle">~2.3 s pause</text>')

# tracking start / end lines, frame bracket, labels
y_lines_bottom = last_base + 48
for tt in (t_start, t_end):
    add(f'<line class="sf-mark" x1="{f(x_of(tt))}" y1="{f(yb + 4)}" x2="{f(x_of(tt))}" y2="{f(y_lines_bottom)}"/>')
yf = last_base + 24
for tt, name in ((t_start, 'tracking start'), (t_end, 'tracking end')):
    add(f'<text class="sf-label-n" x="{f(x_of(tt))}" y="{f(y_lines_bottom + 20)}" text-anchor="middle">{name}</text>')

# 0.5 s scale bar (no end caps)
ys = yf
add(f'<line class="sf-scale" x1="{f(x_of(0.1))}" y1="{f(ys)}" x2="{f(x_of(0.6))}" y2="{f(ys)}"/>')
add(f'<text class="sf-label-n" x="{f(x_of(0.35))}" y="{f(ys + 20)}" text-anchor="middle">0.5 s</text>')

# legend and Triggerbox zoom at the top, side by side
yl = 46
lx = X_LABEL
add(f'<path class="sf-ar sf-step" d="M{f(lx)} {f(yl + 6)}V{f(yl - 8)}H{f(lx + 14)}V{f(yl + 6)}"/>')
add(f'<text class="sf-sub" x="{f(lx + 22)}" y="{f(yl + 4)}">Arduino SYNC pulse: 250 ms on, 250–1500 ms off</text>')
# Triggerbox zoom (true scale, measured on ch04 of 20260421_182605: 0.116 ms high every 6.66 ms)
ZOOM_T0 = 8.8                          # where the zoomed segment sits on the top trace (s, drawn axis)
row0_top = Y_FIRST - TRACE_H / 2
add(f'<rect class="sf-zoombox" x="{f(x_of(ZOOM_T0) - 4)}" y="{f(row0_top - 4)}" width="14" height="{f(TRACE_H + 8)}" rx="2"/>')
ix0, ix1 = 470, W - 2
add(f'<rect class="sf-zoombox" x="{ix0}" y="{f(yl - 38)}" width="{ix1 - ix0}" height="92" rx="3"/>')
# two lines join the boxed segment's corners to the panel's bottom corners
box_x0, box_x1, box_y = x_of(ZOOM_T0) - 4, x_of(ZOOM_T0) + 10, row0_top - 4
add(f'<line class="sf-zoomline" x1="{f(box_x0)}" y1="{f(box_y)}" x2="{ix0}" y2="{f(yl + 54)}"/>')
add(f'<line class="sf-zoomline" x1="{f(box_x1)}" y1="{f(box_y)}" x2="{ix1}" y2="{f(yl + 54)}"/>')
add(f'<text class="sf-sub" x="{ix0 + 12}" y="{f(yl - 18)}">boxed segment, true scale: 0.116 ms on, every 6.66 ms</text>')
z_left, z_right, z_ms = ix0 + 14, ix1 - 14, 26.0
pxms = (z_right - z_left) / z_ms
zb = yl + 34
ztop = zb - 18
edges = [2.0 + 6.66 * k for k in range(4)]
d = f'M{f(z_left)} {f(zb)}'
for e in edges:
    xa, xb2 = z_left + e * pxms, z_left + (e + 0.116) * pxms
    d += f'H{f(xa)}V{f(ztop)}H{f(xb2)}V{f(zb)}'
d += f'H{f(z_right)}'
add(f'<path class="sf-tb sf-step" d="{d}"/>')
xa, xb2 = z_left + edges[1] * pxms, z_left + edges[2] * pxms
add(f'<line class="sf-muted-line" x1="{f(xa + 2)}" y1="{f(yl + 8)}" x2="{f(xb2 - 2)}" y2="{f(yl + 8)}" marker-start="url(#sf-arr)" marker-end="url(#sf-arr)"/>')
add(f'<text class="sf-note" x="{f((xa + xb2) / 2)}" y="{f(yl + 3)}" text-anchor="middle">6.66 ms</text>')
add(f'<line class="sf-scale" x1="{f(z_left)}" y1="{f(yl + 46)}" x2="{f(z_left + 5 * pxms)}" y2="{f(yl + 46)}"/>')
add(f'<text class="sf-label-n" x="{f(z_left + 5 * pxms + 8)}" y="{f(yl + 50)}">5 ms</text>')

H = y_lines_bottom + 30

style = """<style>
.sync-fig { width: 100%; max-width: 960px; height: auto; display: block; margin: 0.5em auto 1.5em; font-family: inherit; }
.sync-fig { --sf-ink: #404040; --sf-muted: #6a6a6a; --sf-none: #c9c9c9; --sf-bg: #fcfcfc; --sf-tb: #5b4bc4; --sf-ar: #c4651a; }
html[data-theme="dark"] .sync-fig { --sf-ink: #c1c1c1; --sf-muted: #8f8f8f; --sf-none: #3a3a3a; --sf-bg: #141414; --sf-tb: #a99cf0; --sf-ar: #f0a25a; }
.sync-fig text { font-family: inherit; }
.sync-fig .sf-label { font-size: 19px; font-weight: 700; fill: var(--sf-ink); }
.sync-fig .sf-label-n { font-size: 17px; fill: var(--sf-ink); }
.sync-fig .sf-sub { font-size: 16px; fill: var(--sf-muted); }
.sync-fig .sf-note { font-size: 16px; fill: var(--sf-muted); font-style: italic; }
.sync-fig .sf-tbtext { fill: var(--sf-tb); }
.sync-fig .sf-tb { stroke: var(--sf-tb); fill: none; }
.sync-fig .sf-ar { stroke: var(--sf-ar); fill: none; }
.sync-fig .sf-base { stroke-width: 1; opacity: 0.55; }
.sync-fig .sf-tick { stroke-width: 1.2; }
.sync-fig .sf-step { stroke-width: 1.6; stroke-linejoin: miter; }
.sync-fig .sf-led { fill: var(--sf-ar); fill-opacity: 0.3; }
.sync-fig .sf-none { stroke: var(--sf-none); stroke-width: 1; stroke-dasharray: 2 3; }
.sync-fig .sf-bg { fill: var(--sf-bg); }
.sync-fig .sf-break { stroke: var(--sf-muted); stroke-width: 1.2; }
.sync-fig .sf-mark { stroke: var(--sf-ink); stroke-width: 1; stroke-dasharray: 4 4; }
.sync-fig .sf-muted-line { stroke: var(--sf-muted); stroke-width: 1; }
.sync-fig .sf-bracket { stroke-width: 1; }
.sync-fig .sf-scale { stroke: var(--sf-ink); stroke-width: 2; }
.sync-fig .sf-arrow-tb { fill: var(--sf-tb); }
.sync-fig .sf-arrow { fill: var(--sf-muted); }
.sync-fig .sf-zoomline { stroke: var(--sf-muted); stroke-width: 1; }
.sync-fig .sf-zoombox { fill: none; stroke: var(--sf-ink); stroke-width: 1.2; stroke-dasharray: 3 2; }
</style>"""

defs = ('<defs>'
        '<marker id="sf-arr-tb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="sf-arrow-tb" d="M0 0L10 5L0 10z"/></marker>'
        '<marker id="sf-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="sf-arrow" d="M0 0L10 5L0 10z"/></marker>'
        '</defs>')
svg = (f'<svg class="sync-fig" viewBox="-4 0 {W + 8} {f(H)}" role="img" '
       'aria-label="Sync pulses received by each recording device: the Loopbio Triggerbox camera-frame pulses and the '
       'Arduino SYNC pulses, with the tracking start after the ~2.3 s Triggerbox pause and the tracking end N pulses later">'
       + defs + ''.join(parts) + '</svg>')
fragment = style + '\n' + svg + '\n'
(out_dir / 'sync_inputs.html').write_text(fragment)

demo = f"""<!doctype html><html><head><meta charset="utf-8"><title>Sync figure demo</title>
<style>
body {{ margin: 0; font-family: Lato, "proxima-nova", "Helvetica Neue", Arial, sans-serif; }}
.pane {{ padding: 24px 36px; }}
.light {{ background: #fcfcfc; color: #404040; }}
.dark {{ background: #141414; color: #c1c1c1; }}
h3 {{ font-family: "Roboto Slab", serif; font-weight: 700; margin: 0 0 8px; }}
p {{ max-width: 900px; line-height: 1.5; }}
</style></head><body>
<div class="pane light"><h3>Light mode</h3><p>Sample paragraph in the page's own font, to compare with the figure text.</p>{fragment}</div>
<div class="pane dark"><h3>Dark mode (default)</h3><p>Sample paragraph in the page's own font, to compare with the figure text.</p>
<div data-dark>{fragment.replace('html[data-theme="dark"] .sync-fig', '[data-dark] .sync-fig')}</div></div>
</body></html>"""
if '--demo' in sys.argv[2:]:
    (out_dir / 'sync_inputs_demo.html').write_text(demo)
