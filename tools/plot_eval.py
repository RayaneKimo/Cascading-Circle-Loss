"""
Gradio app for interactive bar-chart comparison of evaluation JSON files.
Uses Plotly for fully interactive charts (hover, zoom, pan).

Usage:
    python tools/plot_eval.py                          # default: reads <repo>/eval/
    python tools/plot_eval.py --eval_dir /path/to/eval       # custom directory
    python tools/plot_eval.py --port 7860                    # custom port
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")

PLOTLY_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


# ── helpers ──────────────────────────────────────────────────────

def shorten_model_name(filename: str) -> str:
    name = filename.replace(".json", "")
    name = re.sub(r"^qwen06B-neg-10-bs128-lr1e-05-ep-4-", "", name)
    name = re.sub(r"_checkpoint-?", " ckpt-", name)
    name = re.sub(r"checkpoint-?", "ckpt-", name)
    return name


def full_model_name(filename: str) -> str:
    return filename.replace(".json", "")


def load_eval_results(eval_dir: str) -> tuple[dict, dict]:
    """Returns (results, full_names) where full_names maps short->full."""
    results = {}
    full_names = {}
    for f in sorted(Path(eval_dir).glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        retrieval = data.get("retrieval", data)
        label = shorten_model_name(f.name)
        results[label] = retrieval
        full_names[label] = full_model_name(f.name)
    return results, full_names


def collect_metric_families(results: dict):
    all_keys: set = set()
    for metrics in results.values():
        all_keys.update(k for k in metrics if "@" in k)

    families: dict[str, list[str]] = defaultdict(list)
    for key in sorted(all_keys):
        family = key.rsplit("@", 1)[0]
        families[family].append(key)

    for family in families:
        families[family].sort(key=lambda k: int(k.rsplit("@", 1)[1]))
    return families


def ordered_families(families: dict) -> list[str]:
    preferred = ["recall", "mrr", "map", "precision", "graded_ndcg"]
    out = [f for f in preferred if f in families]
    out += [f for f in families if f not in preferred]
    return out


# ── Plotly charts ────────────────────────────────────────────────

def build_comparison_plot(selected_models, selected_families):
    all_results, full_names = load_eval_results(EVAL_DIR)
    if not all_results:
        return go.Figure().add_annotation(text="No JSON files found in eval/",
                                          showarrow=False, font=dict(size=18))

    if not selected_models:
        selected_models = list(all_results.keys())
    model_names = [m for m in all_results if m in selected_models]

    families = collect_metric_families(all_results)
    family_keys = ordered_families(families)
    if selected_families:
        family_keys = [f for f in family_keys if f in selected_families]

    if not family_keys:
        return go.Figure().add_annotation(text="No metric families selected",
                                          showarrow=False, font=dict(size=18))

    n = len(family_keys)
    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f.upper() for f in family_keys],
        vertical_spacing=0.08,
    )

    for row_idx, fam in enumerate(family_keys, start=1):
        metric_keys = families[fam]
        x_labels = [f"@{k.rsplit('@', 1)[1]}" for k in metric_keys]

        for mi, model in enumerate(model_names):
            vals = [all_results[model].get(k, 0.0) for k in metric_keys]
            hover_full = full_names.get(model, model)
            fig.add_trace(
                go.Bar(
                    x=x_labels, y=vals, name=model,
                    marker_color=PLOTLY_COLORS[mi % len(PLOTLY_COLORS)],
                    text=[f"{v:.4f}" for v in vals],
                    textposition="outside", textfont_size=9,
                    customdata=[hover_full] * len(vals),
                    hovertemplate=(
                        "<b>%{customdata}</b><br>"
                        "%{x}: %{y:.4f}<extra></extra>"
                    ),
                    legendgroup=model,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx, col=1,
            )

        fig.update_yaxes(title_text="Score", range=[0, 1.05], row=row_idx, col=1)
        fig.update_xaxes(title_text="k", row=row_idx, col=1)

    fig.update_layout(
        barmode="group",
        height=420 * n + 120,
        margin=dict(t=180),
        title_text="Model Evaluation Comparison",
        title_font_size=20,
        title_y=0.98,
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            font_size=10,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        template="plotly_white",
    )
    return fig


def build_single_model_plot(model_name):
    all_results, full_names = load_eval_results(EVAL_DIR)
    if model_name not in all_results:
        empty = go.Figure().add_annotation(text="Model not found",
                                           showarrow=False, font=dict(size=18))
        return empty, ""

    metrics = all_results[model_name]
    families = collect_metric_families({model_name: metrics})
    family_keys = ordered_families(families)

    n = len(family_keys)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f.upper() for f in family_keys],
        horizontal_spacing=0.06,
    )

    for col_idx, fam in enumerate(family_keys, start=1):
        keys = families[fam]
        x_labels = [f"@{k.rsplit('@', 1)[1]}" for k in keys]
        vals = [metrics.get(k, 0.0) for k in keys]

        fig.add_trace(
            go.Bar(
                x=x_labels, y=vals,
                marker_color=PLOTLY_COLORS[:len(keys)],
                text=[f"{v:.4f}" for v in vals],
                textposition="outside", textfont_size=10,
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        fig.update_yaxes(range=[0, 1.05], row=1, col=col_idx)

    fig.update_layout(
        height=450,
        title_text=full_names.get(model_name, model_name),
        title_font_size=14,
        template="plotly_white",
    )

    scalar_lines = []
    for k, v in metrics.items():
        if "@" not in k:
            scalar_lines.append(f"**{k}**: {v}")
    scalar_md = "\n\n".join(scalar_lines) if scalar_lines else ""

    return fig, scalar_md


def build_table_md(selected_models):
    all_results, _full_names = load_eval_results(EVAL_DIR)
    if not all_results:
        return "No results found."

    if not selected_models:
        selected_models = list(all_results.keys())
    model_names = [m for m in all_results if m in selected_models]

    families = collect_metric_families(all_results)
    family_keys = ordered_families(families)

    all_metric_keys = []
    for fam in family_keys:
        all_metric_keys.extend(families[fam])

    header = "| Model | " + " | ".join(all_metric_keys) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_metric_keys)) + "|"
    rows = []
    for m in model_names:
        vals = [f"{all_results[m].get(k, 0.0):.4f}" for k in all_metric_keys]
        rows.append(f"| {m} | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


def build_summary_table(selected_models, selected_metrics):
    all_results, full_names = load_eval_results(EVAL_DIR)
    if not all_results:
        return "<p>No results found.</p>"

    if not selected_models:
        selected_models = list(all_results.keys())
    model_names = [m for m in all_results if m in selected_models]

    if not selected_metrics:
        return "<p>No metrics selected.</p>"

    def is_ours(name: str) -> bool:
        low = name.lower()
        return "cascl" in low or "cascading" in low

    baseline_models = [m for m in model_names if not is_ours(m)]
    ours_models = [m for m in model_names if is_ours(m)]

    best_per_metric = {}
    for mk in selected_metrics:
        vals = [all_results[m].get(mk, 0.0) for m in model_names]
        best_per_metric[mk] = max(vals) if vals else 0.0

    # Baseline model for relative improvement %
    BASELINE_KEY = "Qwen3-Embedding-0.6B_OTB"
    baseline_metrics = {}
    for m in all_results:
        if "qwen3-embedding" in m.lower() and "otb" in m.lower():
            baseline_metrics = all_results[m]
            break

    css = """
    <style>
      .summary-outer {
        margin: 0; padding: 0; width: 100%;
      }
      .summary-wrap {
        resize: vertical; overflow: auto; min-height: 400px; height: 850px;
        max-height: 2500px; border: 1px solid #ccc; border-radius: 4px;
        padding: 0; background: white; width: 100%; box-sizing: border-box;
      }
      .summary-table {
        width: 100%; border-collapse: collapse; table-layout: fixed;
        font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5px;
      }
      .summary-table th {
        background: #2e7d32; color: white; padding: 6px 2px;
        text-align: center; font-weight: 700; border: 1px solid #1b5e20;
        position: sticky; top: 0; z-index: 2; font-size: 10px;
        white-space: normal; word-break: break-word; line-height: 1.3;
      }
      .summary-table th:first-child { text-align: left; }
      .summary-table td {
        padding: 3px 2px; text-align: center; border: 1px solid #ddd;
        white-space: nowrap; font-size: 10.5px; overflow: hidden;
        text-overflow: ellipsis;
      }
      .summary-table td:first-child {
        text-align: left; font-weight: 500;
        white-space: normal; word-break: break-word;
        line-height: 1.3; padding: 5px 3px;
      }
      .summary-table tr:hover { background: #e8f5e9 !important; }
      .summary-table .best { background: #c8f7c5; color: #1b5e20; font-weight: 700; }
      .summary-table .delta-bad { color: #b71c1c; }
      .summary-table .delta-mid { color: #e65100; }
      .summary-table .delta-ok  { color: #555; }
      .summary-table .sep-row td {
        background: #e0e0e0; text-align: center; font-weight: 700;
        color: #555; padding: 4px; border: 1px solid #ccc;
      }
      .summary-table .ours td { background: #f0faf0; }
      .summary-table .ours td:first-child { font-weight: 700; }
      .summary-table .ours .best { background: #a5e8a5; }
      .summary-table td[title] { cursor: help; }
      .summary-table .drag-handle {
        cursor: grab; user-select: none; padding: 0 2px; color: #aaa;
      }
      .summary-table .drag-handle:active { cursor: grabbing; }
      .summary-table tr.drag-over td { border-top: 3px solid #2e7d32; }
      .summary-table .delta { font-size: 9px; opacity: 0.8; }
      .summary-table .delta-up { color: #1b5e20; }
      .summary-table .baseline-row td { background: #fff3e0; }
      .summary-table .baseline-row td:first-child { font-weight: 700; font-style: italic; }
    </style>
    """

    def metric_cell(m, mk, is_baseline=False):
        val = all_results[m].get(mk, 0.0)
        best = best_per_metric[mk]
        is_best = (best > 0 and val == best)
        full = full_names.get(m, m)

        base_val = baseline_metrics.get(mk, 0.0)
        if base_val > 0 and not is_baseline:
            vs_base = ((val - base_val) / base_val) * 100
            base_str = f"{vs_base:+.1f}% vs OTB"
            base_cls = "delta-up" if vs_base > 0 else "delta-bad" if vs_base < -5 else "delta-mid" if vs_base < 0 else "delta-ok"
        else:
            base_str = "baseline" if is_baseline else ""
            base_cls = "delta-ok"

        if is_best:
            tooltip = f"{full} | {mk}: {val:.4f} | BEST | {base_str}"
            return (f'<td class="best" title="{tooltip}"><b>{val:.4f}</b>'
                    f'<br><span class="delta {base_cls}">{base_str}</span></td>')

        delta_pct = ((val - best) / best) * 100 if best > 0 else 0.0
        tooltip = f"{full} | {mk}: {val:.4f} | vs best: {delta_pct:+.2f}% | {base_str}"
        return (f'<td title="{tooltip}">'
                f'{val:.4f}<br><span class="delta {base_cls}">{base_str}</span></td>')

    # Inline drag-and-drop handlers (Gradio strips <script> tags)
    ROW_DND = (
        ' draggable="true"'
        ' ondragstart="'
            "this.style.opacity=0.4;"
            "this.parentNode._dragRow=this;"
            "event.dataTransfer.effectAllowed='move';"
        '"'
        ' ondragover="'
            "event.preventDefault();"
            "event.dataTransfer.dropEffect='move';"
            "this.classList.add('drag-over');"
        '"'
        ' ondragleave="this.classList.remove(\'drag-over\')"'
        ' ondrop="'
            "event.preventDefault();"
            "this.classList.remove('drag-over');"
            "var dr=this.parentNode._dragRow;"
            "if(dr&&dr!==this){"
              "var rows=Array.from(this.parentNode.children);"
              "if(rows.indexOf(dr)<rows.indexOf(this))this.after(dr);else this.before(dr);"
            "}"
        '"'
        ' ondragend="'
            "this.style.opacity=1;"
            "this.parentNode.querySelectorAll('tr').forEach(function(r){r.classList.remove('drag-over')});"
        '"'
    )

    def model_cell(m, is_ours_flag):
        full = full_names.get(m, m)
        tag = f"<b>{m}</b>" if is_ours_flag else m
        return (f'<td title="{full}">'
                f'<span class="drag-handle">&#9776;</span> {tag}</td>')

    def is_baseline(name: str) -> bool:
        low = name.lower()
        return "qwen3-embedding" in low and "otb" in low

    def make_row(m, ours=False):
        bl = is_baseline(m)
        cls_parts = []
        if ours:
            cls_parts.append("ours")
        if bl:
            cls_parts.append("baseline-row")
        cls = f' class="{" ".join(cls_parts)}"' if cls_parts else ''
        cells = model_cell(m, ours)
        cells += "".join(metric_cell(m, mk, is_baseline=bl) for mk in selected_metrics)
        return f"<tr{ROW_DND}{cls}>{cells}</tr>"

    header = ("<tr><th>Model</th>"
              + "".join(f"<th>{mk}</th>" for mk in selected_metrics)
              + "</tr>")

    sep_row = (
        f'<tr class="sep-row"{ROW_DND}><td colspan="{1 + len(selected_metrics)}">'
        f"CasCL (ours)</td></tr>"
    )

    rows = [make_row(m) for m in baseline_models]
    if ours_models:
        rows.append(sep_row)
        rows.extend(make_row(m, ours=True) for m in ours_models)

    title = ("Summary &mdash; Best in "
             "<span style='color:#2e7d32;font-weight:700'>green</span>. "
             "Each cell shows <span style='color:#1b5e20'>% improvement</span> "
             "vs <span style='color:#e65100;font-weight:600'>Qwen3-Embedding-0.6B OTB</span> baseline. "
             "<em>Hover cell for details. "
             "Drag &#9776; to reorder. Resize from bottom-right.</em>")

    n_metrics = len(selected_metrics)
    model_col_pct = 18
    metric_col_pct = (100 - model_col_pct) / max(n_metrics, 1)
    colgroup = (f'<colgroup><col style="width:{model_col_pct}%">'
                + "".join(f'<col style="width:{metric_col_pct:.2f}%">'
                          for _ in selected_metrics)
                + '</colgroup>')

    html = f"""{css}
    <div class="summary-outer">
    <p style="font-size:12px;margin:0 0 4px 0;">{title}</p>
    <div class="summary-wrap">
    <table class="summary-table">
      {colgroup}
      <thead>{header}</thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    </div>
    </div>
    """
    return html


# ── LaTeX table generator ────────────────────────────────────────

LATEX_METRIC_LABELS = {
    "recall": "R",
    "mrr": "MRR",
    "map": "MAP",
    "precision": "P",
    "graded_ndcg": "nDCG",
}


def _latex_col_header(metric_key: str) -> str:
    family, k = metric_key.rsplit("@", 1)
    prefix = LATEX_METRIC_LABELS.get(family, family)
    return f"{prefix}@{k}"


def _classify_model(name: str) -> str:
    low = name.lower()
    if "qwen3-embedding" in low and "otb" in low:
        return "pretrained"
    if "ablation" in low:
        return "ablation"
    if "cascl" in low or "cascading" in low:
        return "ours"
    return "baseline"


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def generate_latex_table(selected_models: list[str] | None = None,
                         selected_metrics: list[str] | None = None) -> str:
    all_results, full_names = load_eval_results(EVAL_DIR)
    if not all_results:
        return "% No results found."

    if not selected_models:
        selected_models = list(all_results.keys())
    model_names = [m for m in all_results if m in selected_models]

    if not selected_metrics:
        families = collect_metric_families(all_results)
        fam_order = ordered_families(families)
        selected_metrics = []
        for fam in fam_order:
            selected_metrics.extend(families[fam])

    # Find OTB baseline
    baseline_metrics: dict = {}
    for m in all_results:
        if _classify_model(m) == "pretrained":
            baseline_metrics = all_results[m]
            break

    # Find best baseline (non-ours, non-ablation, non-pretrained)
    best_baseline_per_metric: dict[str, float] = {}
    for mk in selected_metrics:
        vals = [all_results[m].get(mk, 0.0) for m in model_names
                if _classify_model(m) == "baseline"]
        best_baseline_per_metric[mk] = max(vals) if vals else 0.0

    # Best and second-best per metric across ALL selected models
    best_per_metric: dict[str, float] = {}
    second_per_metric: dict[str, float] = {}
    for mk in selected_metrics:
        vals = sorted([all_results[m].get(mk, 0.0) for m in model_names],
                      reverse=True)
        best_per_metric[mk] = vals[0] if vals else 0.0
        second_per_metric[mk] = vals[1] if len(vals) > 1 else 0.0

    # Group models
    groups: dict[str, list[str]] = {
        "pretrained": [], "baseline": [], "ablation": [], "ours": []
    }
    for m in model_names:
        groups[_classify_model(m)].append(m)

    n_cols = len(selected_metrics)
    col_spec = "l" + "c" * n_cols
    col_headers = " & ".join(_latex_col_header(mk) for mk in selected_metrics)

    def fmt_val(m: str, mk: str) -> str:
        val = all_results[m].get(mk, 0.0)
        s = f"{val:.4f}"
        if val == best_per_metric[mk] and val > 0:
            s = r"\textbf{" + s + "}"
        elif val == second_per_metric[mk] and val > 0:
            s = r"\underline{" + s + "}"
        return s

    def fmt_delta(val: float, ref: float) -> str:
        if ref > 0:
            d = ((val - ref) / ref) * 100
            return f"{d:+.1f}\\%"
        return "--"

    def model_row(m: str, label: str | None = None) -> str:
        name = _latex_escape(label or m)
        vals = " & ".join(fmt_val(m, mk) for mk in selected_metrics)
        return f"    {name} & {vals} \\\\"

    def delta_row(label: str, ref_metrics: dict[str, float],
                  src_model: str) -> str:
        name = _latex_escape(label)
        cells = []
        for mk in selected_metrics:
            val = all_results[src_model].get(mk, 0.0)
            ref = ref_metrics.get(mk, 0.0)
            cells.append(fmt_delta(val, ref))
        return f"    {name} & " + " & ".join(cells) + " \\\\"

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        f"  \\caption{{Retrieval evaluation results. "
        f"\\textbf{{Bold}} = best, \\underline{{underline}} = second best.}}",
        r"  \label{tab:main_results}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Model & {col_headers} \\\\",
        r"    \midrule",
    ]

    # Pretrained baseline
    for m in groups["pretrained"]:
        lines.append(model_row(m, f"\\textit{{{_latex_escape(m)}}}"))

    # Trained baselines
    if groups["baseline"]:
        if groups["pretrained"]:
            lines.append(r"    \addlinespace")
        for m in groups["baseline"]:
            lines.append(model_row(m))

    # Ablations
    if groups["ablation"]:
        lines.append(r"    \midrule")
        lines.append(r"    \addlinespace")
        lines.append(
            r"    \multicolumn{" + str(n_cols + 1)
            + r"}{l}{\textit{Ablations}} \\"
        )
        lines.append(r"    \addlinespace")
        for m in groups["ablation"]:
            lines.append(model_row(m))

    # CasCL (ours)
    if groups["ours"]:
        lines.append(r"    \midrule")
        for m in groups["ours"]:
            lines.append(model_row(m))

        # Delta rows for best CasCL model
        best_ours = max(
            groups["ours"],
            key=lambda m: sum(all_results[m].get(mk, 0.0)
                              for mk in selected_metrics),
        )

        lines.append(r"    \addlinespace")
        if baseline_metrics:
            lines.append(delta_row(
                r"$\Delta$ vs OTB", baseline_metrics, best_ours))
        if any(v > 0 for v in best_baseline_per_metric.values()):
            lines.append(delta_row(
                r"$\Delta$ vs best baseline",
                best_baseline_per_metric, best_ours))

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


# ── Gradio UI ────────────────────────────────────────────────────

def create_app():
    all_results, _full_names = load_eval_results(EVAL_DIR)
    model_choices = list(all_results.keys())
    families = collect_metric_families(all_results)
    family_choices = list(families.keys())

    fam_order = ordered_families(families)
    all_metric_keys = []
    for fam in fam_order:
        all_metric_keys.extend(families[fam])

    with gr.Blocks(
        title="Eval Dashboard",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 100vw !important; width: 100vw !important; padding: 4px !important; margin: 0 !important; }
            .contain { max-width: 100% !important; }
            .wrap { max-width: 100% !important; }
            #component-0 { max-width: 100% !important; }
            .tabitem { padding: 4px !important; }
            .block { padding: 4px !important; }
        """
    ) as app:
        gr.Markdown("# Embedding Model Evaluation Dashboard")
        gr.Markdown(f"Reading from `{EVAL_DIR}` — **{len(model_choices)} models** found.")

        with gr.Tab("Compare Models"):
            with gr.Row():
                model_selector = gr.CheckboxGroup(
                    choices=model_choices, value=model_choices,
                    label="Select models",
                )
                family_selector = gr.CheckboxGroup(
                    choices=family_choices, value=family_choices,
                    label="Select metric families",
                )
            compare_btn = gr.Button("Generate Comparison", variant="primary")
            compare_plot = gr.Plot()
            compare_btn.click(
                build_comparison_plot,
                inputs=[model_selector, family_selector],
                outputs=compare_plot,
            )

        with gr.Tab("Single Model Detail"):
            single_selector = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                label="Select a model",
            )
            single_btn = gr.Button("Show Details", variant="primary")
            single_plot = gr.Plot()
            single_info = gr.Markdown()
            single_btn.click(
                build_single_model_plot,
                inputs=single_selector,
                outputs=[single_plot, single_info],
            )

        with gr.Tab("Results Table"):
            table_model_selector = gr.CheckboxGroup(
                choices=model_choices, value=model_choices,
                label="Select models",
            )
            table_btn = gr.Button("Generate Table", variant="primary")
            table_md = gr.Markdown()
            table_btn.click(
                build_table_md,
                inputs=table_model_selector,
                outputs=table_md,
            )

        with gr.Tab("LaTeX Export"):
            with gr.Row():
                latex_model_selector = gr.Dropdown(
                    choices=model_choices, value=model_choices,
                    multiselect=True, label="Select models",
                    filterable=True,
                )
                latex_metric_selector = gr.Dropdown(
                    choices=all_metric_keys, value=all_metric_keys,
                    multiselect=True, label="Select metrics",
                    filterable=True,
                )
            latex_btn = gr.Button("Generate LaTeX", variant="primary")
            latex_output = gr.Code(language=None, label="LaTeX Table (copy-paste into your paper)")
            latex_btn.click(
                generate_latex_table,
                inputs=[latex_model_selector, latex_metric_selector],
                outputs=latex_output,
            )

        with gr.Tab("Summary Table"):
            with gr.Row():
                summary_model_selector = gr.Dropdown(
                    choices=model_choices, value=model_choices,
                    multiselect=True, label="Select models",
                    filterable=True,
                )
                summary_metric_selector = gr.Dropdown(
                    choices=all_metric_keys, value=all_metric_keys,
                    multiselect=True, label="Select metrics",
                    filterable=True,
                )
            summary_btn = gr.Button("Generate Summary", variant="primary")
            summary_html = gr.HTML()
            summary_btn.click(
                build_summary_table,
                inputs=[summary_model_selector, summary_metric_selector],
                outputs=summary_html,
            )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eval Dashboard (Gradio)")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table to stdout and exit (no server)")
    args = parser.parse_args()

    if args.eval_dir:
        EVAL_DIR = args.eval_dir

    if args.latex:
        print(generate_latex_table())
    else:
        app = create_app()
        app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
