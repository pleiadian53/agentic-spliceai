"""Generate visual diagram illustrating BPE center-token approximation.

Creates a figure showing:
1. Character-level vs BPE tokenization alignment
2. Why center-token approximation causes positional offset
3. Which token(s) should represent the target nucleotide

Output: bpe-center-token-approximation.png in the same directory.

Usage:
    python generate_bpe_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Colors ──────────────────────────────────────────────────────────────
C_BG = "#fafafa"
C_NT = "#e8e8e8"       # nucleotide cell background
C_TARGET = "#ff6b6b"   # target nucleotide highlight
C_TARGET_LIGHT = "#ffe0e0"
C_BPE = "#4a90d9"      # BPE token
C_BPE_LIGHT = "#d4e6f9"
C_BPE_MATCH = "#27ae60" # BPE token covering target
C_BPE_MATCH_LIGHT = "#d5f5e3"
C_BPE_CENTER = "#f39c12" # center BPE token (wrong one)
C_BPE_CENTER_LIGHT = "#fef3d5"
C_CHAR = "#9b59b6"     # character-level token
C_CHAR_LIGHT = "#ebdef0"
C_ARROW = "#2c3e50"
C_TEXT = "#2c3e50"


def draw_nucleotide_row(ax, y, sequence, target_idx, label, x_start=0):
    """Draw a row of nucleotide cells."""
    for i, nt in enumerate(sequence):
        x = x_start + i
        color = C_TARGET_LIGHT if i == target_idx else C_NT
        edgecolor = C_TARGET if i == target_idx else "#cccccc"
        lw = 2.5 if i == target_idx else 0.8

        rect = mpatches.FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edgecolor, linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, nt, ha="center", va="center",
                fontsize=11, fontfamily="monospace", fontweight="bold",
                color=C_TARGET if i == target_idx else C_TEXT)

    # Row label
    ax.text(x_start - 0.3, y + 0.5, label, ha="right", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)


def draw_token_row(ax, y, tokens, spans, target_nt_idx, label,
                   center_token_idx=None, x_start=0):
    """Draw a row of variable-width tokens with span brackets.

    Returns the index of the token that covers the target nucleotide.
    """
    covering_tok_idx = None
    x = x_start
    for tok_idx, (token_label, span) in enumerate(zip(tokens, spans)):
        # Determine color
        tok_start_nt = sum(spans[:tok_idx])
        tok_end_nt = tok_start_nt + span
        covers_target = tok_start_nt <= target_nt_idx < tok_end_nt

        if covers_target:
            covering_tok_idx = tok_idx
            facecolor = C_BPE_MATCH_LIGHT
            edgecolor = C_BPE_MATCH
        elif tok_idx == center_token_idx:
            facecolor = C_BPE_CENTER_LIGHT
            edgecolor = C_BPE_CENTER
        else:
            facecolor = C_BPE_LIGHT
            edgecolor = C_BPE

        width = span
        rect = mpatches.FancyBboxPatch(
            (x + 0.05, y + 0.05), width - 0.1, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + width / 2, y + 0.5, token_label,
                ha="center", va="center", fontsize=10,
                fontfamily="monospace", fontweight="bold",
                color=edgecolor)
        x += width

    # Row label
    ax.text(x_start - 0.3, y + 0.5, label, ha="right", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)

    return covering_tok_idx


def draw_char_token_row(ax, y, sequence, target_idx, label, x_start=0):
    """Draw character-level tokens (1:1 with nucleotides)."""
    for i, nt in enumerate(sequence):
        x = x_start + i
        is_target = (i == target_idx)
        facecolor = C_CHAR_LIGHT if not is_target else "#d5f5e3"
        edgecolor = C_CHAR if not is_target else C_BPE_MATCH

        rect = mpatches.FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + 0.5, y + 0.5, nt, ha="center", va="center",
                fontsize=10, fontfamily="monospace", fontweight="bold",
                color=edgecolor)

    ax.text(x_start - 0.3, y + 0.5, label, ha="right", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)


def draw_bracket(ax, x_start, x_end, y, label, color, direction="down"):
    """Draw a bracket annotation below/above a span."""
    mid = (x_start + x_end) / 2
    if direction == "down":
        y_base = y - 0.15
        y_tip = y - 0.55
        va = "top"
    else:
        y_base = y + 1.15
        y_tip = y + 1.55
        va = "bottom"

    # Bracket lines
    ax.plot([x_start + 0.2, x_start + 0.2], [y_base, y_tip + 0.1],
            color=color, lw=1.5, clip_on=False)
    ax.plot([x_end - 0.2, x_end - 0.2], [y_base, y_tip + 0.1],
            color=color, lw=1.5, clip_on=False)
    ax.plot([x_start + 0.2, x_end - 0.2], [y_tip + 0.1, y_tip + 0.1],
            color=color, lw=1.5, clip_on=False)

    ax.text(mid, y_tip - 0.05, label, ha="center", va=va,
            fontsize=9, color=color, fontweight="bold")


def draw_arrow(ax, x1, y1, x2, y2, color, label="", style="->"):
    """Draw an annotated arrow."""
    ax.annotate(
        label, xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=2),
        fontsize=9, ha="center", va="center", color=color,
        fontweight="bold",
    )


def main():
    fig, axes = plt.subplots(3, 1, figsize=(18, 16), facecolor=C_BG)
    for ax in axes:
        ax.set_facecolor(C_BG)
        ax.set_xlim(-4.5, 22)
        ax.set_ylim(-2.5, 4.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # ── Example sequence ────────────────────────────────────────────────
    # 20 nucleotides, target is position 10 (0-indexed)
    seq = "ATCGATCGAACGTTAGCTTG"
    target_idx = 10  # the 'G' at position 10

    # BPE tokens (realistic variable-length k-mers)
    bpe_tokens = ["ATC", "GAT", "CGA", "AC", "GTT", "AGC", "TTG"]
    bpe_spans = [3, 3, 3, 2, 3, 3, 3]
    assert sum(bpe_spans) == len(seq)

    # Center BPE token index (naive: len/2)
    n_bpe = len(bpe_tokens)
    center_bpe = n_bpe // 2  # index 3 → "AC" (covers nt 9-10)

    # ════════════════════════════════════════════════════════════════════
    # Panel 1: Character-level models (exact alignment)
    # ════════════════════════════════════════════════════════════════════
    ax = axes[0]
    ax.set_title("Panel A:  Character-Level Models (Evo2, SpliceBERT, HyenaDNA)\n"
                 "Exact 1:1 alignment — center token = target nucleotide",
                 fontsize=13, fontweight="bold", color=C_TEXT, pad=12, loc="left")

    draw_nucleotide_row(ax, 2.5, seq, target_idx, "DNA", x_start=0)
    draw_char_token_row(ax, 0.8, seq, target_idx, "Tokens", x_start=0)

    # Arrow from target nucleotide to target token
    ax.annotate("", xy=(target_idx + 0.5, 1.7), xytext=(target_idx + 0.5, 2.45),
                arrowprops=dict(arrowstyle="->", color=C_BPE_MATCH, lw=2))
    ax.text(target_idx + 0.5, 2.05, "exact\nmatch", ha="center", va="center",
            fontsize=8, color=C_BPE_MATCH, fontweight="bold")

    # Position labels
    for i in range(len(seq)):
        ax.text(i + 0.5, 3.7, str(i), ha="center", va="center",
                fontsize=7, color="#999999", fontfamily="monospace")
    ax.text(len(seq) / 2, 4.2, "nucleotide position (0-indexed)",
            ha="center", fontsize=8, color="#999999")

    # Target marker
    ax.text(target_idx + 0.5, -0.1, "target pos=10",
            ha="center", va="top", fontsize=9, color=C_TARGET,
            fontweight="bold")
    ax.text(target_idx + 0.5, -0.6,
            "emb[10] → exact embedding for this nucleotide",
            ha="center", va="top", fontsize=9, color=C_BPE_MATCH)

    # Formula
    ax.text(-4.2, 1.5, "center_idx =\nemb.shape[0] // 2\n= 20 // 2 = 10",
            fontsize=8, fontfamily="monospace", color=C_TEXT,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc"))

    # ════════════════════════════════════════════════════════════════════
    # Panel 2: BPE (naive center-token — WRONG)
    # ════════════════════════════════════════════════════════════════════
    ax = axes[1]
    ax.set_title("Panel B:  DNABERT-2 BPE — Naive Center-Token (CURRENT, causes offset)\n"
                 "center_idx = num_tokens // 2, but tokens have variable width",
                 fontsize=13, fontweight="bold", color=C_TEXT, pad=12, loc="left")

    draw_nucleotide_row(ax, 2.5, seq, target_idx, "DNA", x_start=0)

    # BPE token row — highlight center token (wrong) and covering token
    covering_idx = draw_token_row(
        ax, 0.8, bpe_tokens, bpe_spans, target_idx, "BPE tokens",
        center_token_idx=center_bpe, x_start=0,
    )

    # Position labels
    for i in range(len(seq)):
        ax.text(i + 0.5, 3.7, str(i), ha="center", va="center",
                fontsize=7, color="#999999", fontfamily="monospace")

    # Show which nucleotide range the center BPE token covers
    center_bpe_start = sum(bpe_spans[:center_bpe])
    center_bpe_end = center_bpe_start + bpe_spans[center_bpe]
    center_bpe_mid = (center_bpe_start + center_bpe_end) / 2

    # Arrow from center token to its nucleotide range
    draw_bracket(ax, center_bpe_start, center_bpe_end, 0.8,
                 f"center token #{center_bpe}\ncovers nt {center_bpe_start}-{center_bpe_end - 1}",
                 C_BPE_CENTER, direction="down")

    # Show the actual covering token
    cover_start = sum(bpe_spans[:covering_idx])
    cover_end = cover_start + bpe_spans[covering_idx]

    draw_bracket(ax, cover_start, cover_end, 2.5,
                 f"token #{covering_idx} covers\ntarget nt {target_idx}",
                 C_BPE_MATCH, direction="up")

    # Offset annotation
    offset = abs(center_bpe_mid - target_idx)
    ax.annotate(
        "", xy=(target_idx + 0.5, 2.45), xytext=(center_bpe_mid, 1.75),
        arrowprops=dict(arrowstyle="<->", color=C_TARGET, lw=2,
                        linestyle="--"),
    )
    ax.text((target_idx + 0.5 + center_bpe_mid) / 2, 2.3,
            f"offset ≈ {offset:.0f} nt",
            ha="center", va="center", fontsize=10, color=C_TARGET,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_TARGET, alpha=0.9))

    # Formula
    ax.text(-4.2, 1.5, "center_idx =\nemb.shape[0] // 2\n= 7 // 2 = 3\n(token \"AC\")",
            fontsize=8, fontfamily="monospace", color=C_BPE_CENTER,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BPE_CENTER_LIGHT,
                      edgecolor=C_BPE_CENTER))

    # ════════════════════════════════════════════════════════════════════
    # Panel 3: BPE with offset mapping (CORRECT)
    # ════════════════════════════════════════════════════════════════════
    ax = axes[2]
    ax.set_title("Panel C:  DNABERT-2 BPE — Offset-Aligned (CORRECT approach)\n"
                 "Use tokenizer offset_mapping to find which token covers target nt",
                 fontsize=13, fontweight="bold", color=C_TEXT, pad=12, loc="left")

    draw_nucleotide_row(ax, 2.5, seq, target_idx, "DNA", x_start=0)

    # BPE token row — only highlight the correct covering token
    draw_token_row(
        ax, 0.8, bpe_tokens, bpe_spans, target_idx, "BPE tokens",
        center_token_idx=None, x_start=0,
    )

    # Position labels
    for i in range(len(seq)):
        ax.text(i + 0.5, 3.7, str(i), ha="center", va="center",
                fontsize=7, color="#999999", fontfamily="monospace")

    # Arrow from target nucleotide to covering token
    cover_mid = (cover_start + cover_end) / 2
    ax.annotate(
        "", xy=(cover_mid, 1.75), xytext=(target_idx + 0.5, 2.45),
        arrowprops=dict(arrowstyle="->", color=C_BPE_MATCH, lw=2.5),
    )
    ax.text((target_idx + 0.5 + cover_mid) / 2 + 0.8, 2.15,
            f"offset_mapping\n→ token #{covering_idx}",
            ha="center", va="center", fontsize=9, color=C_BPE_MATCH,
            fontweight="bold")

    # Show the mapping table
    mapping_text = "offset_mapping:\n"
    cum = 0
    for i, (tok, span) in enumerate(zip(bpe_tokens, bpe_spans)):
        marker = " ◄── target" if i == covering_idx else ""
        mapping_text += f"  #{i} \"{tok}\" → nt [{cum}:{cum + span}]{marker}\n"
        cum += span

    ax.text(-4.2, 1.5, mapping_text.strip(),
            fontsize=7.5, fontfamily="monospace", color=C_TEXT,
            va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_BPE_MATCH))

    # Annotation: correct embedding
    ax.text(target_idx + 0.5, -0.1,
            f"emb[{covering_idx}] → embedding for token \"{bpe_tokens[covering_idx]}\" "
            f"(covers nt {cover_start}-{cover_end - 1}, including target nt {target_idx})",
            ha="center", va="top", fontsize=9, color=C_BPE_MATCH,
            fontweight="bold")

    # ── Legend ───────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor=C_TARGET_LIGHT, edgecolor=C_TARGET,
                       label="Target nucleotide (pos 10)"),
        mpatches.Patch(facecolor=C_BPE_MATCH_LIGHT, edgecolor=C_BPE_MATCH,
                       label="BPE token covering target (correct)"),
        mpatches.Patch(facecolor=C_BPE_CENTER_LIGHT, edgecolor=C_BPE_CENTER,
                       label="Naive center token (wrong)"),
        mpatches.Patch(facecolor=C_BPE_LIGHT, edgecolor=C_BPE,
                       label="Other BPE tokens"),
        mpatches.Patch(facecolor=C_CHAR_LIGHT, edgecolor=C_CHAR,
                       label="Character-level tokens (1:1)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, fancybox=True,
               edgecolor="#cccccc", framealpha=0.95)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    out_path = Path(__file__).parent / "bpe-center-token-approximation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
