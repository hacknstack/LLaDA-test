from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
CUSTOM_MASK_SEGMENT = "custom masks"

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
        Groups default and custom-mask equivalents together.

        Key = document, model, setup path excluding custom-mask/document/mask suffix.
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


def parse_result_dir(path: Path) -> ResultRun | None:
    """
    Supports both default and custom-mask layouts.

    Default examples:
      results/Llama 3 8B/exact/top_k = 40/MITLicense
      results/LLaDA 8B Base/low-confidence remasking/100 Monte Carlo samples/full/MITLicense
      results/LLaDA 8B Base/ELBO/MITLicense

    Custom-mask examples:
      results/LLaDA 8B Base/low-confidence remasking/100 Monte Carlo samples/full/custom masks/MITLicense/1-24,75-100
      results/LLaDA 8B Base/random remasking/20 trajectories samples/custom masks/MITLicense/2,4,6,8
      results/LLaDA 8B Base/ELBO/custom masks/MITLicense/1-50
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
        idx = parts.index(CUSTOM_MASK_SEGMENT)
        if idx + 2 >= len(parts):
            return None
        document = parts[idx + 1]
        custom_mask = parts[idx + 2]
        setup_parts = tuple(parts[1:idx])
    else:
        document = parts[-1]
        custom_mask = None
        setup_parts = tuple(parts[1:-1])

    setup_label = " / ".join(setup_parts) if setup_parts else "default"

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
def discover_text_documents() -> list[str]:
    if not TEXTS_DIR.exists():
        return []
    return sorted([p.stem for p in TEXTS_DIR.glob("*.txt")], key=natural_key)


@st.cache_data(show_spinner=False)
def discover_runs() -> list[ResultRun]:
    if not RESULTS_DIR.exists():
        return []

    runs: list[ResultRun] = []
    for p in RESULTS_DIR.rglob("*"):
        if looks_like_result_dir(p):
            parsed = parse_result_dir(p)
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


def default_runs_for_document_and_model(
    runs: list[ResultRun],
    document: str,
    model: str,
) -> list[ResultRun]:
    return [
        r
        for r in runs
        if r.document == document
        and r.model == model
        and not r.is_custom_mask
    ]


def custom_runs_equivalent_to(runs: list[ResultRun], base_run: ResultRun) -> list[ResultRun]:
    return [
        r
        for r in runs
        if r.document == base_run.document
        and r.model == base_run.model
        and r.setup_parts == base_run.setup_parts
        and r.is_custom_mask
    ]


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

text_docs = discover_text_documents()
runs = discover_runs()

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


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("Add graph")

    document = st.selectbox("1. Document", text_docs, index=0)

    if st.session_state.active_document is None:
        st.session_state.active_document = document

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

    if not models:
        st.warning("No results found for this document.")
        st.stop()

    model = st.selectbox("2. Model θ", models)

    default_runs = default_runs_for_document_and_model(runs, document, model)
    setup_options = sorted(default_runs, key=lambda r: natural_key(r.setup_label))

    if not setup_options:
        st.warning("No non-custom-mask setups found for this model/document.")
        st.stop()

    setup_labels = [r.setup_label for r in setup_options]
    selected_setup_label = st.selectbox("3. Setup", setup_labels)

    base_run = setup_options[setup_labels.index(selected_setup_label)]

    custom_equivalents = custom_runs_equivalent_to(runs, base_run)
    custom_equivalents = sorted(
        custom_equivalents,
        key=lambda r: natural_key(r.custom_mask or ""),
    )

    variant_labels = ["default mask"] + [
        f"custom mask: {r.custom_mask}"
        for r in custom_equivalents
    ]

    selected_variant = st.selectbox("4. Mask variant", variant_labels)

    if selected_variant == "default mask":
        run_to_add = base_run
    else:
        idx = variant_labels.index(selected_variant) - 1
        run_to_add = custom_equivalents[idx]

    if custom_equivalents:
        st.info(f"Found {len(custom_equivalents)} custom-mask equivalent(s) for this setup.")
    else:
        st.caption("No custom-mask equivalents found for this setup.")

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

    st.divider()
    st.header("Graph options")

    y_scale = st.radio("Y axis", ["linear", "log"], horizontal=True)
    x_axis = st.radio("X axis", ["window_index", "char_start"], horizontal=True)

    show_markers = st.checkbox("Show markers", value=False)
    show_threshold = st.checkbox("Show extraction threshold 0.001", value=True)

    st.divider()
    st.header("Discovered")

    st.caption(f"Texts: {len(text_docs)}")
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
                    st.caption(f"custom mask: `{run.custom_mask}`")
                else:
                    st.caption("default mask")

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

    if show_threshold:
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

    if y_scale == "log":
        fig.update_yaxes(type="log")

    clicked = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=700,
        key=(
            f"plot_{document}_{y_scale}_{x_axis}_{len(selected_runs)}_"
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