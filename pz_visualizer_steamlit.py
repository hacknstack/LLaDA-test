from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
import streamlit as st
from streamlit_plotly_events import plotly_events


# -----------------------------
# Configuration
# -----------------------------

ROOT = Path(".")
TEXTS_DIR = ROOT / "texts"
RESULTS_DIR = ROOT / "results"
EXPECTED_FILES = {"summary.json", "windows.csv"}
CUSTOM_MASK_SEGMENT = "custom masks"  # Obsolete layout marker; mask runs are no longer read from this folder.
TIMESTAMP_DIR_RE = re.compile(r"^\d{8}[_-]\d{6}$")
DISCOVERY_CACHE_VERSION = "layout-v3-scatter-20260607"

PRESET_TRACE_COLORS = {
    "Blue": "#1f77b4",
    "Orange": "#ff7f0e",
    "Green": "#2ca02c",
    "Red": "#d62728",
    "Purple": "#9467bd",
}


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class ResultRun:
    path: Path
    document: str
    model: str
    setup_parts: tuple[str, ...]
    setup_label: str
    custom_mask: str | None
    summary_path: Path
    windows_path: Path

    @property
    def is_custom_mask(self) -> bool:
        return self.custom_mask is not None

    @property
    def base_key(self) -> tuple[str, str, tuple[str, ...]]:
        """
        Groups runs that differ only by mask together.

        Key = document, model, setup path excluding the mask/document/data suffix.
        """
        return (self.document, self.model, self.setup_parts)

    @property
    def display_label(self) -> str:
        if self.custom_mask:
            return f"{self.model} / {self.setup_label} / mask={self.custom_mask}"
        return f"{self.model} / {self.setup_label}"


# -----------------------------
# Discovery helpers
# -----------------------------

def natural_key(s: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def looks_like_result_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {p.name for p in path.iterdir() if p.is_file()}
    return EXPECTED_FILES.issubset(files)


def setup_label_from_parts(setup_parts: tuple[str, ...]) -> str:
    return " / ".join(setup_parts) if setup_parts else "default"


def is_llada_model(model: str) -> bool:
    return "llada" in model.lower()


def is_likely_mask_segment(segment: str) -> bool:
    """
    Matches masks such as 1,3,5,7 or 1-10,21-30.
    Avoids treating setup labels such as full or top_k = 40 as masks.
    """
    return bool(
        re.fullmatch(
            r"\s*\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*\s*",
            segment,
        )
    )


def contains_trajectory_segment(parts: tuple[str, ...]) -> bool:
    return any("trajector" in part.lower() for part in parts)


def infer_document_index(parts: tuple[str, ...], text_doc_set: set[str]) -> int | None:
    """
    Return the index of the document segment inside a result path.

    The important rule is that the real document name is the path segment
    matching texts/<document>.txt, even when summary.json/windows.csv are one
    layer below the document folder, such as:

      results/<model>/<setup>/<document>/<timestamp>/windows.csv

    If no exact text-doc segment is found, fall back only for common
    timestamp-like data folders. Otherwise return None so the run is not
    incorrectly assigned to a fake document.
    """
    for idx in range(len(parts) - 1, 0, -1):
        if parts[idx] in text_doc_set:
            return idx

    if len(parts) >= 3 and TIMESTAMP_DIR_RE.fullmatch(parts[-1]):
        return len(parts) - 2

    return None


def parse_result_dir(path: Path, text_doc_set: set[str]) -> ResultRun | None:
    """
    Parses result directories containing summary.json and windows.csv.

    Supported layouts:
      results/<model>/<setup>/<document>
      results/<model>/<setup>/<document>/<ignored-data-folder>

    LLaDA trajectory-mask layout:
      results/LLaDA 8B Base/<remasking>/<trajectory setup>/<mask>/<document>
      results/LLaDA 8B Base/<remasking>/<trajectory setup>/<mask>/<document>/<ignored-data-folder>

    Obsolete custom masks folders are ignored; masks are discovered only as
    implicit LLaDA path segments directly before the document name.
    """
    try:
        rel = path.relative_to(RESULTS_DIR)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 2:
        return None

    model = parts[0]
    summary_path = path / "summary.json"
    windows_path = path / "windows.csv"

    if CUSTOM_MASK_SEGMENT in parts:
        return None

    document_idx = infer_document_index(parts, text_doc_set)
    if document_idx is None or document_idx <= 0 or document_idx >= len(parts):
        return None

    document = parts[document_idx]
    if document not in text_doc_set:
        return None
    before_document = tuple(parts[1:document_idx])

    custom_mask = None
    setup_parts = before_document

    # LLaDA trajectory masks are implicit: the segment immediately before
    # the document name is the mask. There is no default mask for these
    # trajectory runs, so a masked run is discovered directly.
    if is_llada_model(model) and before_document:
        possible_mask = before_document[-1]
        setup_before_mask = before_document[:-1]
        if setup_before_mask and (
            contains_trajectory_segment(setup_before_mask)
            or is_likely_mask_segment(possible_mask)
        ):
            custom_mask = possible_mask
            setup_parts = setup_before_mask

    setup_label = setup_label_from_parts(setup_parts)

    return ResultRun(
        path=path,
        document=document,
        model=model,
        setup_parts=setup_parts,
        setup_label=setup_label,
        custom_mask=custom_mask,
        summary_path=summary_path,
        windows_path=windows_path,
    )


@st.cache_data(show_spinner=False)
def discover_text_documents(cache_version: str) -> list[str]:
    if not TEXTS_DIR.exists():
        return []
    return sorted([p.stem for p in TEXTS_DIR.glob("*.txt")], key=natural_key)


@st.cache_data(show_spinner=False)
def discover_runs(cache_version: str, text_docs: tuple[str, ...]) -> list[ResultRun]:
    if not RESULTS_DIR.exists():
        return []

    text_doc_set = set(text_docs)

    runs: list[ResultRun] = []
    for p in RESULTS_DIR.rglob("*"):
        if looks_like_result_dir(p):
            parsed = parse_result_dir(p, text_doc_set)
            if parsed is not None:
                runs.append(parsed)

    return sorted(runs, key=lambda r: natural_key(str(r.path)))


@st.cache_data(show_spinner=False)
def load_windows_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["window_index", "char_start", "char_end", "p_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in windows.csv: {missing}")

    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce").astype("Int64")
    df["char_start"] = pd.to_numeric(df["char_start"], errors="coerce").astype("Int64")
    df["char_end"] = pd.to_numeric(df["char_end"], errors="coerce").astype("Int64")
    df["p_z"] = pd.to_numeric(df["p_z"], errors="coerce")

    df = df.dropna(subset=["window_index", "char_start", "char_end", "p_z"])

    df["window_index"] = df["window_index"].astype(int)
    df["char_start"] = df["char_start"].astype(int)
    df["char_end"] = df["char_end"].astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_text(document: str) -> str:
    path = TEXTS_DIR / f"{document}.txt"
    return path.read_text(encoding="utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> dict[str, Any]:
    return safe_read_json(Path(path))


# -----------------------------
# Selection helpers
# -----------------------------

def runs_for_document(runs: list[ResultRun], document: str) -> list[ResultRun]:
    return [r for r in runs if r.document == document]


def runs_for_document_and_model(
    runs: list[ResultRun],
    document: str,
    model: str,
) -> list[ResultRun]:
    return [r for r in runs if r.document == document and r.model == model]


def setup_parts_for_document_and_model(
    runs: list[ResultRun],
    document: str,
    model: str,
) -> list[tuple[str, ...]]:
    options = {
        r.setup_parts
        for r in runs
        if r.document == document and r.model == model
    }
    return sorted(options, key=lambda parts: natural_key(setup_label_from_parts(parts)))


def runs_for_setup(
    runs: list[ResultRun],
    document: str,
    model: str,
    setup_parts: tuple[str, ...],
) -> list[ResultRun]:
    return sorted(
        [
            r
            for r in runs
            if r.document == document
            and r.model == model
            and r.setup_parts == setup_parts
        ],
        key=lambda r: natural_key(r.custom_mask or "default"),
    )


def run_lookup_key(run: ResultRun) -> str:
    return str(run.path)


def get_run_by_key(runs: list[ResultRun], key: str) -> ResultRun | None:
    for run in runs:
        if run_lookup_key(run) == key:
            return run
    return None


# -----------------------------
# Color helpers
# -----------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_to_css(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"rgb({r},{g},{b})"


def plotly_color_to_hex(color: str) -> str:
    """
    Converts common Plotly color formats to hex.

    Supports:
      - #rrggbb
      - rgb(r,g,b)
      - rgba(r,g,b,a)
    """
    if color.startswith("#"):
        return color

    nums = [int(x) for x in re.findall(r"\d+", color)[:3]]
    if len(nums) == 3:
        return rgb_to_hex((nums[0], nums[1], nums[2]))

    return "#1f77b4"


def blend_with_white(hex_color: str, amount: float) -> str:
    """
    amount=0 keeps original color; amount=1 becomes white.
    Returns an rgb(...) CSS color.
    """
    r, g, b = hex_to_rgb(hex_color)
    mixed = (
        int(r + (255 - r) * amount),
        int(g + (255 - g) * amount),
        int(b + (255 - b) * amount),
    )
    return rgb_to_css(mixed)


def blend_with_white_hex(hex_color: str, amount: float) -> str:
    """
    amount=0 keeps original color; amount=1 becomes white.
    Returns a hex color usable by st.color_picker and Plotly.
    """
    r, g, b = hex_to_rgb(hex_color)
    mixed = (
        int(r + (255 - r) * amount),
        int(g + (255 - g) * amount),
        int(b + (255 - b) * amount),
    )
    return rgb_to_hex(mixed)


def model_color_map(models: list[str]) -> dict[str, str]:
    palette = pc.qualitative.Plotly + pc.qualitative.Dark24 + pc.qualitative.Light24
    return {
        model: plotly_color_to_hex(palette[i % len(palette)])
        for i, model in enumerate(sorted(models, key=natural_key))
    }


def trace_color_for_run_hex(
    run: ResultRun,
    model_colors: dict[str, str],
    occurrence_index: int,
) -> str:
    base = model_colors.get(run.model, "#1f77b4")
    base = plotly_color_to_hex(base)

    shade_cycle = [0.0, 0.22, 0.38, 0.52, 0.66]
    return blend_with_white_hex(base, shade_cycle[occurrence_index % len(shade_cycle)])


def stable_ui_key(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def get_graph_settings(run_key: str) -> dict[str, Any]:
    if "graph_settings" not in st.session_state:
        st.session_state.graph_settings = {}

    if run_key not in st.session_state.graph_settings:
        st.session_state.graph_settings[run_key] = {
            "visible": True,
            "color": None,
        }

    return st.session_state.graph_settings[run_key]


def is_ar_run(run: ResultRun) -> bool:
    """AR means any selected non-LLaDA run."""
    return not is_llada_model(run.model)


def scatter_ar_vs_llada_possible(selected_runs: list[ResultRun]) -> bool:
    """
    Multi-run scatter is supported when exactly one visible run is AR
    and every other visible run is an LLaDA trajectory/mask run.
    """
    ar_runs = [run for run in selected_runs if is_ar_run(run)]
    llada_runs = [run for run in selected_runs if is_llada_model(run.model)]
    return len(ar_runs) == 1 and len(llada_runs) >= 1 and len(selected_runs) >= 2


def default_color_lookup_for_runs(selected_runs: list[ResultRun]) -> dict[str, str]:
    models_for_colors = sorted({run.model for run in selected_runs}, key=natural_key)
    colors_by_model = model_color_map(models_for_colors)
    model_seen_counts: dict[str, int] = {}
    default_colors: dict[str, str] = {}

    for run in selected_runs:
        occurrence_index = model_seen_counts.get(run.model, 0)
        model_seen_counts[run.model] = occurrence_index + 1
        default_colors[run_lookup_key(run)] = trace_color_for_run_hex(
            run,
            colors_by_model,
            occurrence_index,
        )

    return default_colors


def color_for_run_hex(run: ResultRun, default_colors: dict[str, str]) -> str:
    settings = get_graph_settings(run_lookup_key(run))
    return settings.get("color") or default_colors.get(run_lookup_key(run), "#1f77b4")


# -----------------------------
# Streamlit app state
# -----------------------------

st.set_page_config(page_title="p(z) Result Visualizer", layout="wide")

if "selected_run_keys" not in st.session_state:
    st.session_state.selected_run_keys = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "graph_settings" not in st.session_state:
    st.session_state.graph_settings = {}


# -----------------------------
# Header
# -----------------------------

st.title("p(z) Result Visualizer")
st.caption(
    "Compare sliding-window conditional reconstruction probabilities across models, "
    "decoding schemes, estimators, remasking strategies, and masks."
)

text_docs = discover_text_documents(DISCOVERY_CACHE_VERSION)
runs = discover_runs(DISCOVERY_CACHE_VERSION, tuple(text_docs))
result_documents = {r.document for r in runs}
selectable_documents = [doc for doc in text_docs if doc in result_documents]

if not TEXTS_DIR.exists():
    st.error("Could not find texts/ in the current working directory.")
    st.stop()

if not RESULTS_DIR.exists():
    st.error("Could not find results/ in the current working directory.")
    st.stop()

if not text_docs:
    st.error("No .txt files found in texts/.")
    st.stop()

if not runs:
    st.error(
        "No result directories found under results/. "
        "Expected summary.json and windows.csv in each run directory."
    )
    st.stop()

if not selectable_documents:
    st.error(
        "No documents in texts/ have corresponding result runs. "
        "Documents with no results are hidden from the selection."
    )
    st.caption(
        f"Discovery debug: {len(text_docs)} text document(s), "
        f"{len(runs)} parsed result run(s), "
        f"{len(result_documents)} parsed result document name(s)."
    )
    with st.expander("Parsed result document names"):
        st.write(sorted(result_documents, key=natural_key))
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("Add graph")

    if st.session_state.active_document not in selectable_documents:
        st.session_state.active_document = selectable_documents[0]

    default_document_index = selectable_documents.index(st.session_state.active_document)
    document = st.selectbox(
        "1. Document",
        selectable_documents,
        index=default_document_index,
    )

    if document != st.session_state.active_document:
        previous_selected_runs = [
            get_run_by_key(runs, key)
            for key in st.session_state.selected_run_keys
        ]
        previous_selected_runs = [r for r in previous_selected_runs if r is not None]

        carried_keys: list[str] = []
        for old_run in previous_selected_runs:
            matching_new_run = next(
                (
                    r
                    for r in runs
                    if r.document == document
                    and r.model == old_run.model
                    and r.setup_parts == old_run.setup_parts
                    and r.custom_mask == old_run.custom_mask
                ),
                None,
            )
            if matching_new_run is not None:
                new_key = run_lookup_key(matching_new_run)
                old_key = run_lookup_key(old_run)

                carried_keys.append(new_key)

                if old_key in st.session_state.graph_settings:
                    st.session_state.graph_settings[new_key] = st.session_state.graph_settings[old_key]

        st.session_state.active_document = document
        st.session_state.selected_run_keys = carried_keys

    doc_runs = runs_for_document(runs, document)
    models = sorted({r.model for r in doc_runs}, key=natural_key)

    model = st.selectbox("2. Model θ", models)

    setup_options = setup_parts_for_document_and_model(runs, document, model)

    if not setup_options:
        st.warning("No setups found for this model/document.")
        st.stop()

    setup_labels = [setup_label_from_parts(parts) for parts in setup_options]
    selected_setup_label = st.selectbox("3. Setup", setup_labels)
    selected_setup_parts = setup_options[setup_labels.index(selected_setup_label)]

    setup_runs = runs_for_setup(runs, document, model, selected_setup_parts)
    default_variants = [r for r in setup_runs if not r.is_custom_mask]
    mask_variants = [r for r in setup_runs if r.is_custom_mask]

    variant_runs: list[ResultRun] = []
    variant_labels: list[str] = []

    for run in default_variants:
        variant_runs.append(run)
        variant_labels.append("no mask")

    for run in mask_variants:
        variant_runs.append(run)
        variant_labels.append(f"mask: {run.custom_mask}")

    if not variant_runs:
        st.warning("No runs found for this setup.")
        st.stop()

    selected_variant = st.selectbox("4. Mask", variant_labels)
    run_to_add = variant_runs[variant_labels.index(selected_variant)]

    if mask_variants and not default_variants:
        st.info(
            "This setup has LLaDA trajectory masks only. "
            "Select a mask to add a trajectory run."
        )
    elif mask_variants:
        st.info(f"Found {len(mask_variants)} mask variant(s) for this setup.")
    else:
        st.caption("This setup has no mask variants.")

    add_col, clear_col = st.columns(2)

    with add_col:
        if st.button("Add", use_container_width=True):
            key = run_lookup_key(run_to_add)
            if key not in st.session_state.selected_run_keys:
                st.session_state.selected_run_keys.append(key)
                get_graph_settings(key)
            else:
                st.warning("That graph is already added.")

    with clear_col:
        if st.button("Clear", use_container_width=True):
            for key in st.session_state.selected_run_keys:
                st.session_state.graph_settings.pop(key, None)
            st.session_state.selected_run_keys = []

    if "plot_mode" not in st.session_state:
        st.session_state.plot_mode = "line"
    if "scatter_axis_choice" not in st.session_state:
        st.session_state.scatter_axis_choice = 0

    selected_graphs = [
        get_run_by_key(runs, key)
        for key in st.session_state.selected_run_keys
    ]
    selected_graphs = [
        r
        for r in selected_graphs
        if r is not None and r.document == document
    ]
    visible_selected_graphs = [
        r
        for r in selected_graphs
        if get_graph_settings(run_lookup_key(r)).get("visible", True)
    ]

    st.divider()
    st.header("Graph options")

    y_scale = st.radio("Y axis", ["linear", "log"], horizontal=True)
    x_scale = st.radio("X axis scale", ["linear", "log"], horizontal=True)

    scatter_pairwise_available = len(visible_selected_graphs) == 2
    scatter_ar_llada_available = (
        len(visible_selected_graphs) > 2
        and scatter_ar_vs_llada_possible(visible_selected_graphs)
    )
    scatter_disabled = not (scatter_pairwise_available or scatter_ar_llada_available)

    plot_mode = st.radio(
        "Plot mode",
        ["line", "scatter"],
        index=1 if st.session_state.plot_mode == "scatter" else 0,
        horizontal=True,
        disabled=scatter_disabled,
    )
    if plot_mode == "scatter" and scatter_disabled:
        plot_mode = "line"
    st.session_state.plot_mode = plot_mode

    x_axis = "window_index"
    if plot_mode != "scatter":
        x_axis = st.radio("X axis", ["window_index", "char_start"], horizontal=True)
    elif scatter_ar_llada_available:
        ar_run = next(run for run in visible_selected_graphs if is_ar_run(run))
        llada_runs = [run for run in visible_selected_graphs if is_llada_model(run.model)]
        st.info(
            "Scatter mode compares one AR trajectory against all visible LLaDA trajectories. "
            "Each LLaDA trajectory is listed separately and keeps its own point color."
        )
    else:
        st.info("Scatter mode displays p_z values from two visible graphs against each other.")

    show_markers = st.checkbox("Show markers", value=False)
    label = (
        "Show y=x reference line"
        if plot_mode == "scatter"
        else "Show extraction threshold 0.001"
    )
    show_reference_line = st.checkbox(
        label,
        value=True,
    )

    if plot_mode == "scatter":
        if scatter_ar_llada_available:
            ar_run = next(run for run in visible_selected_graphs if is_ar_run(run))
            llada_runs = [run for run in visible_selected_graphs if is_llada_model(run.model)]
            scatter_axis_labels = [
                f"AR: {ar_run.display_label} → x, LLaDA trajectories → y",
                f"LLaDA trajectories → x, AR: {ar_run.display_label} → y",
            ]
            selected_axis_label = st.radio(
                "Scatter axes",
                scatter_axis_labels,
                index=min(st.session_state.scatter_axis_choice, len(scatter_axis_labels) - 1),
            )
            st.session_state.scatter_axis_choice = scatter_axis_labels.index(selected_axis_label)

            with st.expander("LLaDA trajectories in scatter"):
                for run in llada_runs:
                    st.caption(run.display_label)
        elif len(visible_selected_graphs) == 2:
            scatter_axis_labels = [
                f"{visible_selected_graphs[0].display_label} → x, {visible_selected_graphs[1].display_label} → y",
                f"{visible_selected_graphs[1].display_label} → x, {visible_selected_graphs[0].display_label} → y",
            ]
            selected_axis_label = st.radio(
                "Scatter axes",
                scatter_axis_labels,
                index=min(st.session_state.scatter_axis_choice, len(scatter_axis_labels) - 1),
            )
            st.session_state.scatter_axis_choice = scatter_axis_labels.index(selected_axis_label)

    st.divider()
    st.header("Discovered")

    st.caption(f"Texts: {len(text_docs)}")
    st.caption(f"Documents with results: {len(selectable_documents)}")
    st.caption(f"Result runs: {len(runs)}")


# -----------------------------
# Main layout
# -----------------------------

selected_runs = [
    get_run_by_key(runs, key)
    for key in st.session_state.selected_run_keys
]
selected_runs = [
    r
    for r in selected_runs
    if r is not None and r.document == document
]

left, right = st.columns([3, 1], gap="large")

with right:
    st.subheader("Added graphs")

    if not selected_runs:
        st.info("Add at least one graph from the sidebar.")
    else:
        for i, run in enumerate(selected_runs):
            run_key = run_lookup_key(run)
            safe_key = stable_ui_key(run_key)
            settings = get_graph_settings(run_key)

            with st.container(border=True):
                st.markdown(f"**{i + 1}. {run.model}**")
                st.caption(run.setup_label)

                if run.custom_mask:
                    st.caption(f"mask: `{run.custom_mask}`")
                else:
                    st.caption("no mask")

                settings["visible"] = st.toggle(
                    "Visible",
                    value=bool(settings.get("visible", True)),
                    key=f"visible_{safe_key}",
                )

                st.caption("Preset colors")

                preset_cols = st.columns(5)

                for col, (color_name, color_hex) in zip(
                    preset_cols,
                    PRESET_TRACE_COLORS.items(),
                ):
                    with col:
                        st.markdown(
                            f"""
                            <div style="
                                width: 100%;
                                height: 28px;
                                border-radius: 8px;
                                background: {color_hex};
                                border: 1px solid rgba(128,128,128,0.4);
                                margin-bottom: 0.25rem;
                            " title="{color_name}"></div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "Set",
                            key=f"preset_{safe_key}_{color_name}",
                            help=color_name,
                            use_container_width=True,
                        ):
                            settings["color"] = color_hex
                            st.rerun()

                current_color = settings.get("color") or "#1f77b4"

                settings["color"] = st.color_picker(
                    "Custom color",
                    value=current_color,
                    key=f"custom_color_{safe_key}",
                )

                if st.button(
                    "Remove",
                    key=f"remove_{safe_key}",
                    use_container_width=True,
                ):
                    st.session_state.selected_run_keys.remove(run_key)
                    st.session_state.graph_settings.pop(run_key, None)
                    st.rerun()

        st.download_button(
            "Download selected run paths",
            data="\n".join(run_lookup_key(r) for r in selected_runs),
            file_name=f"selected_runs_{document}.txt",
            mime="text/plain",
            use_container_width=True,
        )


with left:
    st.subheader(f"Document: {document}")

    if not selected_runs:
        st.stop()

    fig = go.Figure()

    loaded_frames: list[tuple[ResultRun, pd.DataFrame]] = []
    visible_loaded_frames: list[tuple[ResultRun, pd.DataFrame]] = []
    errors: list[str] = []

    for run in selected_runs:
        try:
            df = load_windows_csv(str(run.windows_path))
            loaded_frames.append((run, df))

            run_key = run_lookup_key(run)
            settings = get_graph_settings(run_key)

            if settings.get("visible", True):
                visible_loaded_frames.append((run, df))

        except Exception as exc:
            errors.append(f"{run.display_label}: {exc}")

    for error in errors:
        st.error(error)

    if not visible_loaded_frames:
        st.info("All added graphs are currently hidden. Toggle at least one graph visible to display it.")
        st.stop()

    x_run: ResultRun | None = None
    y_run: ResultRun | None = None
    ar_run_for_scatter: ResultRun | None = None

    visible_runs_for_scatter = [run for run, _ in visible_loaded_frames]
    scatter_ar_llada_mode = (
        st.session_state.plot_mode == "scatter"
        and len(visible_loaded_frames) > 2
        and scatter_ar_vs_llada_possible(visible_runs_for_scatter)
    )
    scatter_pairwise_mode = (
        st.session_state.plot_mode == "scatter"
        and len(visible_loaded_frames) == 2
    )
    scatter_mode = scatter_pairwise_mode or scatter_ar_llada_mode

    if scatter_mode:
        default_colors = default_color_lookup_for_runs(visible_runs_for_scatter)

        if scatter_pairwise_mode:
            axis_order = st.session_state.scatter_axis_choice
            x_run, x_df = visible_loaded_frames[axis_order]
            y_run, y_df = visible_loaded_frames[1 - axis_order]

            trace_color = color_for_run_hex(x_run, default_colors)

            merged_df = pd.merge(
                x_df[["window_index", "char_start", "char_end", "p_z"]],
                y_df[["window_index", "p_z"]],
                on="window_index",
                how="inner",
                suffixes=("_x", "_y"),
            )

            if x_scale == "log":
                merged_df = merged_df[merged_df["p_z_x"] > 0].copy()
            if y_scale == "log":
                merged_df = merged_df[merged_df["p_z_y"] > 0].copy()

            if merged_df.empty:
                st.error("Scatter mode could not align the two selected runs by window_index with the requested axis scales.")
                st.stop()

            customdata = merged_df[
                ["window_index", "char_start", "char_end", "p_z_x", "p_z_y"]
            ].to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=merged_df["p_z_x"],
                    y=merged_df["p_z_y"],
                    mode="markers",
                    name=f"{x_run.display_label} vs {y_run.display_label}",
                    customdata=customdata,
                    marker=dict(color=trace_color, size=8),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        f"x: {x_run.display_label}<br>"
                        f"y: {y_run.display_label}<br>"
                        "window_index: %{customdata[0]}<br>"
                        "chars: %{customdata[1]}-%{customdata[2]}<br>"
                        "p_z_x: %{customdata[3]:.6g}<br>"
                        "p_z_y: %{customdata[4]:.6g}<extra></extra>"
                    ),
                )
            )

            line_min = float(min(merged_df["p_z_x"].min(), merged_df["p_z_y"].min()))
            line_max = float(max(merged_df["p_z_x"].max(), merged_df["p_z_y"].max()))
            xaxis_title = f"p_z ({x_run.display_label})"
            yaxis_title = f"p_z ({y_run.display_label})"

        else:
            frame_by_key = {run_lookup_key(run): (run, df) for run, df in visible_loaded_frames}
            ar_frames = [(run, df) for run, df in visible_loaded_frames if is_ar_run(run)]
            llada_frames = [
                (run, df)
                for run, df in visible_loaded_frames
                if is_llada_model(run.model)
            ]

            ar_run_for_scatter, ar_df = ar_frames[0]
            ar_on_x = st.session_state.scatter_axis_choice == 0

            all_scatter_values: list[float] = []
            aligned_any = False

            for llada_run, llada_df in llada_frames:
                merged_df = pd.merge(
                    ar_df[["window_index", "char_start", "char_end", "p_z"]],
                    llada_df[["window_index", "p_z"]],
                    on="window_index",
                    how="inner",
                    suffixes=("_ar", "_llada"),
                )

                if ar_on_x:
                    x_values = merged_df["p_z_ar"]
                    y_values = merged_df["p_z_llada"]
                    x_label = f"AR: {ar_run_for_scatter.display_label}"
                    y_label = "LLaDA trajectories"
                else:
                    x_values = merged_df["p_z_llada"]
                    y_values = merged_df["p_z_ar"]
                    x_label = "LLaDA trajectories"
                    y_label = f"AR: {ar_run_for_scatter.display_label}"

                plot_df = merged_df.copy()
                plot_df["scatter_x"] = x_values
                plot_df["scatter_y"] = y_values

                if x_scale == "log":
                    plot_df = plot_df[plot_df["scatter_x"] > 0].copy()
                if y_scale == "log":
                    plot_df = plot_df[plot_df["scatter_y"] > 0].copy()

                if plot_df.empty:
                    continue

                aligned_any = True
                all_scatter_values.extend(plot_df["scatter_x"].astype(float).tolist())
                all_scatter_values.extend(plot_df["scatter_y"].astype(float).tolist())

                plot_df["ar_run_key"] = run_lookup_key(ar_run_for_scatter)
                plot_df["llada_run_key"] = run_lookup_key(llada_run)

                customdata = plot_df[
                    [
                        "window_index",
                        "char_start",
                        "char_end",
                        "p_z_ar",
                        "p_z_llada",
                        "ar_run_key",
                        "llada_run_key",
                    ]
                ].to_numpy()

                trace_color = color_for_run_hex(llada_run, default_colors)

                fig.add_trace(
                    go.Scatter(
                        x=plot_df["scatter_x"],
                        y=plot_df["scatter_y"],
                        mode="markers",
                        name=llada_run.display_label,
                        customdata=customdata,
                        marker=dict(color=trace_color, size=8),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            f"AR: {ar_run_for_scatter.display_label}<br>"
                            "window_index: %{customdata[0]}<br>"
                            "chars: %{customdata[1]}-%{customdata[2]}<br>"
                            "p_z_AR: %{customdata[3]:.6g}<br>"
                            "p_z_LLaDA: %{customdata[4]:.6g}<extra></extra>"
                        ),
                    )
                )

            if not aligned_any:
                st.error("Scatter mode could not align the AR run with any visible LLaDA trajectory by window_index with the requested axis scales.")
                st.stop()

            line_min = float(min(all_scatter_values))
            line_max = float(max(all_scatter_values))
            xaxis_title = f"p_z ({x_label})"
            yaxis_title = f"p_z ({y_label})"

        if show_reference_line:
            if x_scale == "log" or y_scale == "log":
                positive_values = [value for value in [line_min, line_max] if value > 0]
                positive_min = min(positive_values) if positive_values else np.nextafter(0, 1)
                positive_max = max(positive_values) if positive_values else 1.0
                if positive_max <= positive_min:
                    line_x = np.array([positive_min, positive_max])
                else:
                    line_x = np.geomspace(positive_min, positive_max, num=100)
            elif line_max <= line_min:
                line_x = np.array([line_min, line_max])
            else:
                line_x = np.linspace(line_min, line_max, num=100)
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_x,
                    mode="lines",
                    name="y=x",
                    line=dict(color="#888888", dash="dash"),
                    showlegend=True,
                )
            )

        fig.update_layout(
            height=700,
            margin=dict(l=90, r=40, t=60, b=90),
            xaxis_title=dict(text=xaxis_title, standoff=22),
            yaxis_title=dict(text=yaxis_title, standoff=28),
            legend_title="Runs",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )
    else:
        all_models_for_colors = sorted(
            {r.model for r, _ in visible_loaded_frames},
            key=natural_key,
        )

        colors_by_model = model_color_map(all_models_for_colors)
        model_seen_counts: dict[str, int] = {}

        for run, df in visible_loaded_frames:
            mode = "lines+markers" if show_markers else "lines"

            occurrence_index = model_seen_counts.get(run.model, 0)
            model_seen_counts[run.model] = occurrence_index + 1

            default_trace_color = trace_color_for_run_hex(
                run,
                colors_by_model,
                occurrence_index,
            )

            settings = get_graph_settings(run_lookup_key(run))
            trace_color = settings.get("color") or default_trace_color

            if y_scale == "log":
                # Plotly log axes cannot display zero.
                # Keep raw p_z in customdata and show only positive values.
                plot_df = df[df["p_z"] > 0].copy()
            else:
                plot_df = df.copy()

            if x_scale == "log":
                plot_df = plot_df[plot_df[x_axis] > 0].copy()

            customdata = plot_df[
                ["window_index", "char_start", "char_end", "p_z"]
            ].to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=plot_df[x_axis],
                    y=plot_df["p_z"],
                    mode=mode,
                    name=run.display_label,
                    customdata=customdata,
                    line=dict(color=trace_color, width=2.5),
                    marker=dict(color=trace_color, size=6),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        f"{x_axis}: %{{x}}<br>"
                        "window_index: %{customdata[0]}<br>"
                        "chars: %{customdata[1]}-%{customdata[2]}<br>"
                        "p_z: %{customdata[3]:.6g}<extra></extra>"
                    ),
                )
            )

        if show_reference_line:
            fig.add_hline(
                y=0.001,
                line_dash="dash",
                annotation_text="0.001 threshold",
                annotation_position="top left",
            )

        fig.update_layout(
            height=700,
            margin=dict(l=90, r=40, t=60, b=90),
            xaxis_title=dict(text=x_axis, standoff=22),
            yaxis_title=dict(text="p_z", standoff=28),
            legend_title="Runs",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

    fig.update_xaxes(
        automargin=True,
        ticks="outside",
        ticklabelstandoff=8,
        title_standoff=22,
    )

    fig.update_yaxes(
        automargin=True,
        ticks="outside",
        ticklabelstandoff=8,
        title_standoff=28,
    )

    if x_scale == "log":
        fig.update_xaxes(type="log")
    if y_scale == "log":
        fig.update_yaxes(type="log")

    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=700,
        key=(
            f"plot_{document}_{y_scale}_{x_scale}_{x_axis}_{len(selected_runs)}_"
            f"{st.session_state.plot_mode}_{st.session_state.scatter_axis_choice}_"
            f"{hash(str(st.session_state.graph_settings))}"
        ),
    )

    st.caption("Click a point on the graph to inspect the exact character chunk used for that window.")

    # -----------------------------
    # Clicked-window inspection
    # -----------------------------

    if clicked:
        point = clicked[0]
        curve_number = point.get("curveNumber")
        customdata = point.get("customdata")

        if customdata is not None and curve_number is not None:
            if scatter_mode:
                if scatter_pairwise_mode and len(customdata) == 5:
                    window_index, char_start, char_end, p_z_x, p_z_y = customdata
                    window_label = f"{x_run.display_label} vs {y_run.display_label}" if x_run and y_run else "scatter trace"

                    window_index = int(window_index)
                    char_start = int(char_start)
                    char_end = int(char_end)
                    p_z_x = float(p_z_x)
                    p_z_y = float(p_z_y)

                    full_text = load_text(document)
                    chunk = full_text[char_start:char_end]

                    st.divider()
                    st.subheader("Clicked window")

                    c1, c2, c3, c4, c5 = st.columns(5)

                    c1.metric("window_index", window_index)
                    c2.metric("char_start", char_start)
                    c3.metric("char_end", char_end)
                    c4.metric("p_z_x", f"{p_z_x:.6g}")
                    c5.metric("p_z_y", f"{p_z_y:.6g}")

                    with st.container(border=True):
                        st.markdown("**Run**")
                        st.write(window_label)
                        if x_run:
                            st.code(str(x_run.path), language="text")
                        if y_run:
                            st.code(str(y_run.path), language="text")

                    st.markdown("**Text chunk used before tokenization**")

                    st.text_area(
                        "chunk",
                        value=chunk,
                        height=260,
                        label_visibility="collapsed",
                    )
                elif scatter_ar_llada_mode and len(customdata) == 7:
                    (
                        window_index,
                        char_start,
                        char_end,
                        p_z_ar,
                        p_z_llada,
                        ar_run_key,
                        llada_run_key,
                    ) = customdata

                    ar_run_clicked = get_run_by_key(runs, str(ar_run_key))
                    llada_run_clicked = get_run_by_key(runs, str(llada_run_key))

                    window_index = int(window_index)
                    char_start = int(char_start)
                    char_end = int(char_end)
                    p_z_ar = float(p_z_ar)
                    p_z_llada = float(p_z_llada)

                    full_text = load_text(document)
                    chunk = full_text[char_start:char_end]

                    st.divider()
                    st.subheader("Clicked window")

                    c1, c2, c3, c4, c5 = st.columns(5)

                    c1.metric("window_index", window_index)
                    c2.metric("char_start", char_start)
                    c3.metric("char_end", char_end)
                    c4.metric("p_z_AR", f"{p_z_ar:.6g}")
                    c5.metric("p_z_LLaDA", f"{p_z_llada:.6g}")

                    with st.container(border=True):
                        st.markdown("**Runs**")
                        if ar_run_clicked is not None:
                            st.write(f"AR: {ar_run_clicked.display_label}")
                            st.code(str(ar_run_clicked.path), language="text")
                        if llada_run_clicked is not None:
                            st.write(f"LLaDA: {llada_run_clicked.display_label}")
                            st.code(str(llada_run_clicked.path), language="text")

                    st.markdown("**Text chunk used before tokenization**")

                    st.text_area(
                        "chunk",
                        value=chunk,
                        height=260,
                        label_visibility="collapsed",
                    )
                else:
                    st.warning("Clicked item does not correspond to a data trace.")
            else:
                # If threshold hline or another shape gets clicked,
                # curveNumber may not map to visible_loaded_frames.
                if 0 <= curve_number < len(visible_loaded_frames):
                    run, _ = visible_loaded_frames[curve_number]

                    window_index, char_start, char_end, p_z = customdata

                    window_index = int(window_index)
                    char_start = int(char_start)
                    char_end = int(char_end)
                    p_z = float(p_z)

                    full_text = load_text(document)
                    chunk = full_text[char_start:char_end]

                    st.divider()
                    st.subheader("Clicked window")

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric("window_index", window_index)
                    c2.metric("char_start", char_start)
                    c3.metric("char_end", char_end)
                    c4.metric("p_z", f"{p_z:.6g}")

                    with st.container(border=True):
                        st.markdown("**Run**")
                        st.write(run.display_label)
                        st.code(str(run.path), language="text")

                    st.markdown("**Text chunk used before tokenization**")

                    st.text_area(
                        "chunk",
                        value=chunk,
                        height=260,
                        label_visibility="collapsed",
                    )
                else:
                    st.warning("Clicked item does not correspond to a data trace.")
    else:
        st.divider()
        st.subheader("Clicked window")
        st.info("No point selected yet.")


# -----------------------------
# Optional run details
# -----------------------------

with st.expander("Selected run summaries"):
    if not selected_runs:
        st.write("No runs selected.")
    else:
        for run in selected_runs:
            summary = load_summary(str(run.summary_path))

            st.markdown(f"### {run.display_label}")
            st.code(str(run.path), language="text")

            if summary:
                compact = {
                    "num_windows_total": summary.get("num_windows_total"),
                    "num_windows_scored": summary.get("num_windows_scored"),
                    "num_windows_extracted": summary.get("num_windows_extracted"),
                    "extraction_rate": summary.get("extraction_rate"),
                    "parameters": summary.get("parameters"),
                    "p_z_distribution": summary.get("p_z_distribution"),
                }

                st.json(compact)
            else:
                st.warning("Could not read summary.json")