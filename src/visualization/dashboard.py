"""
visualization/dashboard.py — Self-contained HTML report generator.
Produces a single file with verdict table, per-hypothesis sections,
and embedded charts — clean enough to share with a non-technical reader.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.hypotheses.runner import HypothesisResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _verdict_badge(verdict: str) -> str:
    colors = {"SUPPORTED": "#2e7d32", "FAILED": "#c62828", "INCONCLUSIVE": "#e65100"}
    color = colors.get(verdict, "#555")
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
        f'background:{color};color:#fff;font-weight:bold;font-size:0.85em">{verdict}</span>'
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_html_report(
    results: dict[str, HypothesisResult],
    charts: dict[str, plt.Figure],
    output_path: Path,
) -> None:
    """
    Write a self-contained HTML report.

    Args:
        results:  mapping of hypothesis_id → HypothesisResult
        charts:   mapping of hypothesis_id → matplotlib Figure
        output_path: destination .html file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Embed charts as base64
    chart_b64: dict[str, str] = {}
    for hid, fig in charts.items():
        if fig is not None:
            chart_b64[hid] = _fig_to_b64(fig)

    # Sort results by hypothesis id numerically
    def _sort_key(hid: str) -> int:
        try:
            return int("".join(filter(str.isdigit, hid)))
        except ValueError:
            return 999

    ordered = sorted(results.values(), key=lambda r: _sort_key(r.hypothesis_id))

    # Verdict summary counts
    supported = sum(1 for r in ordered if r.verdict.value == "SUPPORTED")
    failed = sum(1 for r in ordered if r.verdict.value == "FAILED")
    inconclusive = sum(1 for r in ordered if r.verdict.value == "INCONCLUSIVE")

    # Build verdict table rows
    table_rows = ""
    for r in ordered:
        table_rows += f"""
        <tr>
          <td style="font-weight:bold;color:#333">{r.hypothesis_id}</td>
          <td>{r.title}</td>
          <td style="text-align:center">{_verdict_badge(r.verdict.value)}</td>
          <td style="font-size:0.82em;color:#555">{r.primary_metric}</td>
          <td style="font-size:0.82em;text-align:right">{r.primary_value:,.3f}</td>
        </tr>"""

    # Build per-hypothesis sections
    sections = ""
    for r in ordered:
        hid = r.hypothesis_id
        chart_html = ""
        if hid in chart_b64:
            chart_html = (
                f'<img src="data:image/png;base64,{chart_b64[hid]}" '
                f'style="max-width:100%;border-radius:6px;margin-top:12px" />'
            )

        verdict_colors = {"SUPPORTED": "#e8f5e9", "FAILED": "#ffebee", "INCONCLUSIVE": "#fff3e0"}
        bg = verdict_colors.get(r.verdict.value, "#fafafa")

        sections += f"""
        <div style="margin:32px 0;padding:24px;background:{bg};border-radius:10px;
                    border-left:5px solid {'#2e7d32' if r.verdict.value=='SUPPORTED' else '#c62828' if r.verdict.value=='FAILED' else '#e65100'}">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
            <span style="font-size:1.3em;font-weight:bold;color:#222">{hid}</span>
            {_verdict_badge(r.verdict.value)}
          </div>
          <h3 style="margin:0 0 10px;font-size:1em;color:#333">{r.title}</h3>
          <p style="margin:0 0 8px;font-size:0.92em;line-height:1.55;color:#444">{r.key_finding}</p>
          <div style="font-size:0.80em;color:#666;margin-top:4px">
            <strong>Primary metric:</strong> {r.primary_metric} &nbsp;|&nbsp;
            <strong>Value:</strong> {r.primary_value:,.3f}
            {"&nbsp;|&nbsp;<strong>p-value:</strong> " + f"{r.p_value:.4f}" if r.p_value is not None else ""}
          </div>
          {chart_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OT Visibility Model — Full Results</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #f4f4f0; color: #222; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 32px 24px; }}
    h1 {{ font-size: 1.6em; font-weight: 800; margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 0.9em; margin-bottom: 28px; }}
    .summary-bar {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
    .summary-card {{ flex: 1; min-width: 120px; padding: 16px; border-radius: 10px;
                     text-align: center; }}
    .summary-card .number {{ font-size: 2em; font-weight: 800; }}
    .summary-card .label {{ font-size: 0.8em; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff;
             border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    th {{ background: #333; color: #fff; padding: 10px 12px; text-align: left; font-size: 0.85em; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f9f9f7; }}
    .footer {{ margin-top: 48px; font-size: 0.78em; color: #999; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>OT Visibility Model — Full Results</h1>
    <p class="subtitle">
      Monte Carlo simulation: 10 competing hypotheses tested against Meridian Precision Manufacturing
      (820 OT assets, 40% baseline inventory coverage) — testing whether "OT visibility is overrated."
    </p>

    <div class="summary-bar">
      <div class="summary-card" style="background:#e8f5e9;color:#2e7d32">
        <div class="number">{supported}</div>
        <div class="label">SUPPORTED</div>
      </div>
      <div class="summary-card" style="background:#ffebee;color:#c62828">
        <div class="number">{failed}</div>
        <div class="label">FAILED</div>
      </div>
      <div class="summary-card" style="background:#fff3e0;color:#e65100">
        <div class="number">{inconclusive}</div>
        <div class="label">INCONCLUSIVE</div>
      </div>
      <div class="summary-card" style="background:#e3f2fd;color:#1565c0">
        <div class="number">{len(ordered)}</div>
        <div class="label">TOTAL</div>
      </div>
    </div>

    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Hypothesis</th>
          <th style="text-align:center">Verdict</th>
          <th>Primary Metric</th>
          <th style="text-align:right">Value</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>

    {sections}

    <div class="footer">
      Generated by the OT Visibility Model &mdash;
      Monte Carlo simulation, {len(ordered)} hypotheses &mdash;
      Meridian Precision Manufacturing synthetic dataset
    </div>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
