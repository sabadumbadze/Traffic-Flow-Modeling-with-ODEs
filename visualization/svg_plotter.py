"""Pure-Python SVG chart generator for scientific plotting (no matplotlib needed)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Series:
    name: str
    x: List[float]
    y: List[float]
    color: str
    stroke_width: float = 2.0
    dashed: bool = False
    marker: Optional[str] = None  # "circle", "square", None


class SVGPlot:
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        width: int = 900,
        height: int = 550,
        log_x: bool = False,
        log_y: bool = False,
    ) -> None:
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.width = width
        self.height = height
        self.log_x = log_x
        self.log_y = log_y
        self.series: List[Series] = []
        self.margin_left = 90
        self.margin_right = 40
        self.margin_top = 60
        self.margin_bottom = 70

    def add_series(
        self,
        name: str,
        x: List[float],
        y: List[float],
        color: str,
        stroke_width: float = 2.0,
        dashed: bool = False,
        marker: Optional[str] = None,
    ) -> None:
        self.series.append(Series(name, list(x), list(y), color, stroke_width, dashed, marker))

    def render(self) -> str:
        all_x = [v for s in self.series for v in s.x if not self.log_x or v > 0]
        all_y = [v for s in self.series for v in s.y if not self.log_y or v > 0]
        if not all_x or not all_y:
            return "<svg></svg>"

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        if not self.log_x:
            span_x = max(1e-9, max_x - min_x)
            min_x -= 0.02 * span_x
            max_x += 0.02 * span_x
        if not self.log_y:
            span_y = max(1e-9, max_y - min_y)
            min_y = max(0.0, min_y - 0.05 * span_y) if min_y >= 0 else min_y - 0.05 * span_y
            max_y += 0.05 * span_y

        plot_w = self.width - self.margin_left - self.margin_right
        plot_h = self.height - self.margin_top - self.margin_bottom

        def transform_x(val: float) -> float:
            if self.log_x:
                lx = math.log10(max(1e-12, val))
                lmin = math.log10(min_x)
                lmax = math.log10(max_x)
                return self.margin_left + (lx - lmin) / max(1e-9, lmax - lmin) * plot_w
            return self.margin_left + (val - min_x) / max(1e-9, max_x - min_x) * plot_w

        def transform_y(val: float) -> float:
            if self.log_y:
                ly = math.log10(max(1e-12, val))
                lmin = math.log10(min_y)
                lmax = math.log10(max_y)
                return self.margin_top + plot_h - (ly - lmin) / max(1e-9, lmax - lmin) * plot_h
            return self.margin_top + plot_h - (val - min_y) / max(1e-9, max_y - min_y) * plot_h

        elements: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" width="{self.width}" height="{self.height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
            f'<text x="{self.width/2}" y="35" text-anchor="middle" font-size="18" font-weight="bold" fill="#1e293b">{self.title}</text>',
            f'<rect x="{self.margin_left}" y="{self.margin_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>',
        ]

        # Grid lines and Ticks
        n_ticks = 6
        if self.log_x:
            lo_pow = math.floor(math.log10(min_x))
            hi_pow = math.ceil(math.log10(max_x))
            for p in range(lo_pow, hi_pow + 1):
                val = 10.0**p
                if min_x <= val <= max_x:
                    px = transform_x(val)
                    elements.append(f'<line x1="{px}" y1="{self.margin_top}" x2="{px}" y2="{self.margin_top+plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
                    elements.append(f'<text x="{px}" y="{self.margin_top+plot_h+20}" text-anchor="middle" font-size="12" fill="#64748b">10^{p}</text>')
        else:
            for i in range(n_ticks + 1):
                val = min_x + i * (max_x - min_x) / n_ticks
                px = transform_x(val)
                elements.append(f'<line x1="{px}" y1="{self.margin_top}" x2="{px}" y2="{self.margin_top+plot_h}" stroke="#e2e8f0" stroke-width="1"/>')
                elements.append(f'<text x="{px}" y="{self.margin_top+plot_h+20}" text-anchor="middle" font-size="12" fill="#64748b">{val:.2f}</text>')

        if self.log_y:
            lo_pow = math.floor(math.log10(min_y))
            hi_pow = math.ceil(math.log10(max_y))
            for p in range(lo_pow, hi_pow + 1):
                val = 10.0**p
                if min_y <= val <= max_y:
                    py = transform_y(val)
                    elements.append(f'<line x1="{self.margin_left}" y1="{py}" x2="{self.margin_left+plot_w}" y2="{py}" stroke="#e2e8f0" stroke-width="1"/>')
                    elements.append(f'<text x="{self.margin_left-10}" y="{py+4}" text-anchor="end" font-size="12" fill="#64748b">10^{p}</text>')
        else:
            for i in range(n_ticks + 1):
                val = min_y + i * (max_y - min_y) / n_ticks
                py = transform_y(val)
                elements.append(f'<line x1="{self.margin_left}" y1="{py}" x2="{self.margin_left+plot_w}" y2="{py}" stroke="#e2e8f0" stroke-width="1"/>')
                elements.append(f'<text x="{self.margin_left-10}" y="{py+4}" text-anchor="end" font-size="12" fill="#64748b">{val:.2f}</text>')

        # Axis Titles
        elements.append(f'<text x="{self.margin_left + plot_w/2}" y="{self.height - 20}" text-anchor="middle" font-size="14" font-weight="600" fill="#334155">{self.xlabel}</text>')
        elements.append(f'<text transform="rotate(-90)" x="{-self.margin_top - plot_h/2}" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="#334155">{self.ylabel}</text>')

        # Render Series Lines
        for s in self.series:
            pts = []
            for vx, vy in zip(s.x, s.y):
                if (not self.log_x or vx > 0) and (not self.log_y or vy > 0):
                    pts.append(f"{transform_x(vx):.1f},{transform_y(vy):.1f}")
            if pts:
                dash_attr = ' stroke-dasharray="6,4"' if s.dashed else ''
                elements.append(f'<polyline fill="none" stroke="{s.color}" stroke-width="{s.stroke_width}"{dash_attr} points="{" ".join(pts)}"/>')

            if s.marker == "circle":
                for vx, vy in zip(s.x, s.y):
                    if (not self.log_x or vx > 0) and (not self.log_y or vy > 0):
                        elements.append(f'<circle cx="{transform_x(vx):.1f}" cy="{transform_y(vy):.1f}" r="4" fill="{s.color}" stroke="#ffffff" stroke-width="1.5"/>')
            elif s.marker == "square":
                for vx, vy in zip(s.x, s.y):
                    if (not self.log_x or vx > 0) and (not self.log_y or vy > 0):
                        px, py = transform_x(vx), transform_y(vy)
                        elements.append(f'<rect x="{px-3.5:.1f}" y="{py-3.5:.1f}" width="7" height="7" fill="{s.color}" stroke="#ffffff" stroke-width="1.5"/>')

        # Legend Box
        if len(self.series) > 1 or self.series:
            leg_x = self.margin_left + 15
            leg_y = self.margin_top + 15
            line_h = 22
            leg_w = max(140, max(len(s.name) * 8 for s in self.series) + 50)
            leg_h = len(self.series) * line_h + 10
            elements.append(f'<rect x="{leg_x}" y="{leg_y}" width="{leg_w}" height="{leg_h}" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" stroke-width="1" rx="4"/>')
            for idx, s in enumerate(self.series):
                ly = leg_y + 16 + idx * line_h
                dash_attr = ' stroke-dasharray="4,2"' if s.dashed else ''
                elements.append(f'<line x1="{leg_x+10}" y1="{ly-4}" x2="{leg_x+35}" y2="{ly-4}" stroke="{s.color}" stroke-width="2.5"{dash_attr}/>')
                if s.marker == "circle":
                    elements.append(f'<circle cx="{leg_x+22.5}" cy="{ly-4}" r="3" fill="{s.color}"/>')
                elif s.marker == "square":
                    elements.append(f'<rect x="{leg_x+19.5}" y="{ly-7}" width="6" height="6" fill="{s.color}"/>')
                elements.append(f'<text x="{leg_x+45}" y="{ly}" font-size="12" fill="#334155">{s.name}</text>')

        elements.append("</svg>")
        return "\n".join(elements)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render())
