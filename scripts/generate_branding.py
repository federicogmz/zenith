#!/usr/bin/env python3
"""
Zenith Branding Generator
Generates:
  - docs/assets/logo.svg (840x240 wide banner, dark card #0b1017, crispEdges pixel-art)
  - docs/assets/icon.svg (128x128 favicon square card)
  - docs/assets/logo_ascii.txt (terminal ASCII/Unicode block-art <= 76 cols)
"""

import html
import math
import os
import sys

# Color Palette
BG_COLOR = "#0b1017"
CARD_BORDER = "#1e293b"
GRID_LINE = "#132130"

# DEM Hypsometric colors (teal / slate / deep navy)
DEM_PALETTE = [
    "#081420",  # 0: deep abyss
    "#0b1c2b",  # 1: very low
    "#0f273b",  # 2: low
    "#13354c",  # 3: mid-low
    "#17435f",  # 4: mid
    "#1b5272",  # 5: mid-high
    "#216388",  # 6: high
    "#2a77a1",  # 7: peak
]

# Footprint Highlights
FRAME_A_STROKE = "#00e5ff"  # Electric cyan
FRAME_A_TINT = "#004759"

FRAME_B_STROKE = "#ffb300"  # Warm amber
FRAME_B_TINT = "#5a3e00"

OVERLAP_BG = "#00e676"      # Bright emerald / mint
OVERLAP_A_TINT = "#00965d"  # Left of Voronoi seam
OVERLAP_B_TINT = "#43a047"  # Right of Voronoi seam
VORONOI_SEAM = "#ffffff"    # Pure white laser seamline
VORONOI_GLOW = "#b9f6ca"    # Soft glow around seamline

# 5x7 Bitmap Font for ZENITH
BITMAP_FONT_5X7 = {
    'Z': [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "11111",
    ],
    'E': [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    'N': [
        "10001",
        "11001",
        "10101",
        "10011",
        "10011",
        "10001",
        "10001",
    ],
    'I': [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ],
    'T': [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    'H': [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
}


def compute_dem_matrix(grid_size=24):
    """Generates a 24x24 matrix of simulated elevation cells."""
    matrix = []
    for r in range(grid_size):
        row = []
        for c in range(grid_size):
            # Synthetic 2D elevation surface with realistic valley & ridge
            val = (
                math.sin(r * 0.28) * 2.2
                + math.cos(c * 0.24) * 2.0
                + math.sin((r + c) * 0.18) * 1.8
                + math.cos((r - c) * 0.15) * 1.2
            )
            # Normalize to 0..7
            val_norm = int(round((val + 7.2) / 14.4 * 7))
            val_norm = max(0, min(7, val_norm))
            row.append(val_norm)
        matrix.append(row)
    return matrix


def get_cell_color(r, c, dem_val, grid_size=24):
    """
    Computes cell color based on DEM and camera footprints:
    - Frame A: cols 3..14, rows 4..19
    - Frame B: cols 9..20, rows 4..19
    - Overlap: cols 9..14, rows 4..19
    - Voronoi seam: diagonal cut through overlap
    """
    in_a = (4 <= r <= 19) and (3 <= c <= 14)
    in_b = (4 <= r <= 19) and (9 <= c <= 20)
    in_overlap = in_a and in_b

    is_a_border = in_a and (r == 4 or r == 19 or c == 3 or (c == 14 and not in_b))
    is_b_border = in_b and (r == 4 or r == 19 or c == 20 or (c == 9 and not in_a))

    # Overlap specific logic
    if in_overlap:
        # Diagonal Voronoi cut: e.g. from (col 9, row 6) to (col 14, row 17)
        # Seam equation: row ≈ 2.2 * (col - 9) + 6 => row - 2.2*(col - 9) - 6 ≈ 0
        seam_r = 6.0 + 2.2 * (c - 9)
        dist_to_seam = r - seam_r

        if abs(dist_to_seam) < 0.65:
            return VORONOI_SEAM, 1.0  # Crisp Voronoi seamline pixel
        elif abs(dist_to_seam) < 1.3:
            return VORONOI_GLOW, 0.95

        # Border of overlap region
        if r == 4 or r == 19:
            return OVERLAP_BG, 1.0

        if dist_to_seam < 0:
            # Upper-left side of overlap (closer to Frame A centroid)
            base_col = OVERLAP_A_TINT
            opacity = 0.85 + (dem_val / 35.0)
            return base_col, opacity
        else:
            # Lower-right side of overlap (closer to Frame B centroid)
            base_col = OVERLAP_B_TINT
            opacity = 0.85 + (dem_val / 35.0)
            return base_col, opacity

    # Outside overlap: Frame A border & interior
    if is_a_border:
        return FRAME_A_STROKE, 1.0
    if in_a:
        # Subtle cyan hypsometric tint
        tints = ["#04222c", "#07303d", "#0a3d4d", "#0e4b5f", "#115970", "#146882", "#187795", "#1c86a7"]
        return tints[dem_val], 0.95

    # Outside overlap: Frame B border & interior
    if is_b_border:
        return FRAME_B_STROKE, 1.0
    if in_b:
        # Subtle amber hypsometric tint
        tints = ["#261902", "#362304", "#472e06", "#573909", "#69450c", "#7a5110", "#8c5e14", "#9e6b18"]
        return tints[dem_val], 0.95

    # Base DEM background
    return DEM_PALETTE[dem_val], 0.9


def generate_emblem_svg(start_x, start_y, cell_size, grid_size=24, show_hud=True):
    """Renders the 24x24 pixel-art DEM + overlapping footprints as SVG elements."""
    elements = []
    matrix = compute_dem_matrix(grid_size)

    # Frame footprint HUD indicators
    w = grid_size * cell_size
    h = grid_size * cell_size

    # Subtle grid shadow / outer border
    elements.append(
        f'<rect x="{start_x - 3}" y="{start_y - 3}" width="{w + 6}" height="{h + 6}" '
        f'fill="#050a10" stroke="#162333" stroke-width="1.5" rx="3" />'
    )

    # Render each cell
    for r in range(grid_size):
        for c in range(grid_size):
            cx = start_x + c * cell_size
            cy = start_y + r * cell_size
            color, opac = get_cell_color(r, c, matrix[r][c], grid_size)
            # Crisp edges rect with 0.5px gap for discrete raster feel
            elements.append(
                f'<rect x="{cx}" y="{cy}" width="{cell_size - 1}" height="{cell_size - 1}" '
                f'fill="{color}" fill-opacity="{opac:.2f}" shape-rendering="crispEdges"/>'
            )

    # Tech HUD corner brackets & labels if enabled
    if show_hud:
        # Corner brackets
        b_len = 6
        b_col = "#2a4365"
        # Top-left
        elements.append(f'<path d="M{start_x - 6},{start_y + b_len} L{start_x - 6},{start_y - 6} L{start_x + b_len},{start_y - 6}" stroke="{b_col}" stroke-width="1.5" fill="none"/>')
        # Top-right
        elements.append(f'<path d="M{start_x + w - b_len},{start_y - 6} L{start_x + w + 6},{start_y - 6} L{start_x + w + 6},{start_y + b_len}" stroke="{b_col}" stroke-width="1.5" fill="none"/>')
        # Bottom-left
        elements.append(f'<path d="M{start_x - 6},{start_y + h - b_len} L{start_x - 6},{start_y + h + 6} L{start_x + b_len},{start_y + h + 6}" stroke="{b_col}" stroke-width="1.5" fill="none"/>')
        # Bottom-right
        elements.append(f'<path d="M{start_x + w - b_len},{start_y + h + 6} L{start_x + w + 6},{start_y + h + 6} L{start_x + w + 6},{start_y + h - b_len}" stroke="{b_col}" stroke-width="1.5" fill="none"/>')

        # Footprint labels below emblem
        label_y = start_y + h + 18
        elements.append(
            f'<text x="{start_x}" y="{label_y}" fill="#00e5ff" font-family="ui-monospace, monospace" font-size="8.5" font-weight="bold" letter-spacing="0.5">■ FRAME A</text>'
        )
        elements.append(
            f'<text x="{start_x + 60}" y="{label_y}" fill="#00e676" font-family="ui-monospace, monospace" font-size="8.5" font-weight="bold" letter-spacing="0.5">■ OVERLAP</text>'
        )
        elements.append(
            f'<text x="{start_x + 124}" y="{label_y}" fill="#ffb300" font-family="ui-monospace, monospace" font-size="8.5" font-weight="bold" letter-spacing="0.5">■ FRAME B</text>'
        )

    return "\n    ".join(elements)


def generate_pixel_text(word, start_x, start_y, pixel_size=7, letter_spacing=2):
    """
    Renders bitmap font characters as discrete SVG <rect> blocks.
    Applies an electric cyan -> emerald color gradient across the columns.
    """
    elements = []
    curr_x = start_x

    # Calculate total columns for gradient interpolation
    total_cols = len(word) * 5 + (len(word) - 1) * letter_spacing
    global_col = 0

    # Colors for gradient: Electric Cyan (#00e5ff) -> Mint Emerald (#00e676)
    def interpolate_color(t):
        r1, g1, b1 = 0, 229, 255
        r2, g2, b2 = 0, 230, 118
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for char in word:
        glyph = BITMAP_FONT_5X7.get(char.upper())
        if not glyph:
            curr_x += 5 * pixel_size + letter_spacing * pixel_size
            continue

        for r_idx, row in enumerate(glyph):
            for c_idx, bit in enumerate(row):
                if bit == '1':
                    px = curr_x + c_idx * pixel_size
                    py = start_y + r_idx * pixel_size
                    t = (global_col + c_idx) / max(1, total_cols - 1)
                    color = interpolate_color(t)

                    # Pixel block with soft bevel / crisp edge
                    elements.append(
                        f'<rect x="{px}" y="{py}" width="{pixel_size - 1}" height="{pixel_size - 1}" '
                        f'fill="{color}" rx="0.5" shape-rendering="crispEdges"/>'
                    )

        curr_x += (5 + letter_spacing) * pixel_size
        global_col += 5 + letter_spacing

    return "\n    ".join(elements)


def generate_status_pills(start_x, start_y):
    """Renders status badges for Pure Rust, Rayon SIMD, < 2 min Triage."""
    pills = [
        ("[PURE RUST]", "#00e5ff", "#002a33"),
        ("[RAYON PARALLEL]", "#00e676", "#003319"),
        ("[< 2 MIN TRIAGE]", "#ffb300", "#332400"),
        ("[MULTI-CRS]", "#38bdf8", "#082f49"),
        ("[BROWNE-CONRADY]", "#a78bfa", "#2e1065"),
    ]

    elements = []
    curr_x = start_x
    for text, border_col, bg_col in pills:
        char_len = len(text)
        width = char_len * 6.2 + 14
        height = 18

        # Badge pill rect
        elements.append(
            f'<rect x="{curr_x:.1f}" y="{start_y}" width="{width:.1f}" height="{height}" '
            f'fill="{bg_col}" stroke="{border_col}" stroke-width="1" rx="3" shape-rendering="crispEdges"/>'
        )
        # Badge text (properly XML escaped)
        escaped_text = html.escape(text)
        elements.append(
            f'<text x="{curr_x + width / 2:.1f}" y="{start_y + 12.5}" fill="{border_col}" '
            f'font-family="ui-monospace, monospace" font-size="8.5" font-weight="bold" '
            f'text-anchor="middle" letter-spacing="0.4">{escaped_text}</text>'
        )
        curr_x += width + 7

    return "\n    ".join(elements)


def generate_logo_svg(filepath):
    """Generates the primary docs/assets/logo.svg banner (840x240)."""
    width = 840
    height = 240

    # Left Emblem: 24x24 raster, cell_size = 6.5px => 156x156 px
    emblem_svg = generate_emblem_svg(start_x=34, start_y=30, cell_size=6.5, grid_size=24, show_hud=True)

    # Wordmark ZENITH: pixel_size = 8px => 6 letters * 5 + 5 * 2 = 40 cols * 8 = 320px wide, 7 * 8 = 56px high
    title_svg = generate_pixel_text("ZENITH", start_x=244, start_y=36, pixel_size=8, letter_spacing=2)

    # Status Pills
    pills_svg = generate_status_pills(start_x=244, start_y=166)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <!-- Background grid pattern -->
    <pattern id="bg-grid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="{GRID_LINE}" stroke-width="0.75"/>
    </pattern>
    <!-- Subtle glow filter for header elements -->
    <filter id="cyan-glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>

  <!-- Dark Card Container (100% Legible in GitHub Light & Dark Themes) -->
  <rect width="{width}" height="{height}" rx="10" fill="{BG_COLOR}" stroke="{CARD_BORDER}" stroke-width="2"/>
  <rect width="{width}" height="{height}" rx="10" fill="url(#bg-grid)" opacity="0.6"/>

  <!-- Left: 24x24 Raster DEM + Overlapping Footprints Emblem -->
  <g id="emblem">
    {emblem_svg}
  </g>

  <!-- Right: Wordmark ZENITH in Discrete Pixel Matrix -->
  <g id="wordmark" filter="url(#cyan-glow)">
    {title_svg}
  </g>

  <!-- Subtitle -->
  <g id="subtitle">
    <text x="244" y="118" fill="#e2e8f0" font-family="ui-monospace, monospace" font-size="13" font-weight="bold" letter-spacing="1.2">
      2.5D DIRECT ORTHORECTIFICATION &amp; RADIOMETRIC ENGINE
    </text>
    <text x="244" y="140" fill="#64748b" font-family="ui-monospace, monospace" font-size="11" font-weight="normal" letter-spacing="0.8">
      PHOTOGRAMMETRIC CONTRACT LEVEL 0 &amp; LEVEL 1 // TACTICAL UAV RESPONSE // v1.0.0
    </text>
  </g>

  <!-- Status Pills -->
  <g id="status-pills">
    {pills_svg}
  </g>

  <!-- Subtle corner tech accents -->
  <path d="M 820 16 L 828 16 L 828 24" fill="none" stroke="#2563eb" stroke-width="1.5"/>
  <path d="M 820 224 L 828 224 L 828 216" fill="none" stroke="#2563eb" stroke-width="1.5"/>
</svg>
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated: {filepath} ({len(svg_content)} bytes)")


def generate_icon_svg(filepath):
    """Generates the square docs/assets/icon.svg favicon (128x128)."""
    size = 128
    grid_size = 24
    cell_size = 4.2  # 24 * 4.2 = 100.8px
    start_x = (size - grid_size * cell_size) / 2
    start_y = (size - grid_size * cell_size) / 2

    emblem_svg = generate_emblem_svg(start_x=start_x, start_y=start_y, cell_size=cell_size, grid_size=grid_size, show_hud=False)

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">
  <defs>
    <pattern id="icon-grid" width="10" height="10" patternUnits="userSpaceOnUse">
      <path d="M 10 0 L 0 0 0 10" fill="none" stroke="{GRID_LINE}" stroke-width="0.5"/>
    </pattern>
  </defs>

  <!-- Square Card Container -->
  <rect width="{size}" height="{size}" rx="18" fill="{BG_COLOR}" stroke="{CARD_BORDER}" stroke-width="2"/>
  <rect width="{size}" height="{size}" rx="18" fill="url(#icon-grid)" opacity="0.5"/>

  <!-- Centered 24x24 Raster DEM + Overlapping Footprints -->
  <g id="icon-emblem">
    {emblem_svg}
  </g>

  <!-- Tech Corner Markers -->
  <path d="M 10 20 L 10 10 L 20 10" fill="none" stroke="#00e5ff" stroke-width="1.5"/>
  <path d="M 118 20 L 118 10 L 108 10" fill="none" stroke="#ffb300" stroke-width="1.5"/>
  <path d="M 10 108 L 10 118 L 20 118" fill="none" stroke="#00e5ff" stroke-width="1.5"/>
  <path d="M 118 108 L 118 118 L 108 118" fill="none" stroke="#ffb300" stroke-width="1.5"/>
</svg>
"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated: {filepath} ({len(svg_content)} bytes)")


def generate_logo_ascii(filepath):
    """Generates docs/assets/logo_ascii.txt (terminal banner <= 76 cols)."""
    banner = """
┌────────────────────────────────────────────────────────────────────────┐
│  ███████╗███████╗███╗   ██╗██╗████████╗██╗  ██╗   ZENITH PHOTOGRAMMETRY│
│  ╚══███╔╝██╔════╝████╗  ██║██║╚══██╔══╝██║  ██║   v1.0.0 (Pure Rust)   │
│    ███╔╝ █████╗  ██╔██╗ ██║██║   ██║   ███████║   Tactical 2.5D Engine │
│   ███╔╝  ██╔══╝  ██║╚██╗██║██║   ██║   ██╔══██║   < 2 min Thermal L0   │
│  ███████╗███████╗██║ ╚████║██║   ██║   ██║  ██║   < 3 min Direct Ortho │
│  ╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝   Multi-CRS EPSG:9377  │
├────────────────────────────────────────────────────────────────────────┤
│  OVERLAPPING SENSOR FOOTPRINTS & VORONOI SEAMLINE GEOMETRY:            │
│                                                                        │
│        ┌────── FRAME A (Cyan) ──────┐                                  │
│        │  . . . . . . . ┌───────────┴──────────┐                       │
│        │  . . . . . . . │ ░░░░░░░░░ ╲  ▒▒▒▒▒▒▒ │  FRAME B (Amber)      │
│        │  . . . . . . . │ ░░░░░░░░░  ╲ ▒▒▒▒▒▒▒ │                       │
│        └────────────────┤ ░░░░░░░░░   ╲▒▒▒▒▒▒▒ │                       │
│                         └───────────┬──────────┘                       │
│                                     ▲ Voronoi Cut Line                 │
│                                [A∩B] OVERLAP BAND                      │
│                                                                        │
│  • Brown-Conrady Dewarping • Rayon Data Parallelism   • LRF Vertical Datum│
└────────────────────────────────────────────────────────────────────────┘
""".strip()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(banner + "\n")
    print(f"Generated: {filepath} ({len(banner.splitlines())} lines)")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_assets = os.path.join(repo_root, "docs", "assets")

    logo_path = os.path.join(docs_assets, "logo.svg")
    icon_path = os.path.join(docs_assets, "icon.svg")
    ascii_path = os.path.join(docs_assets, "logo_ascii.txt")

    generate_logo_svg(logo_path)
    generate_icon_svg(icon_path)
    generate_logo_ascii(ascii_path)
    print("All branding assets generated successfully.")


if __name__ == "__main__":
    main()
