"""
Genera 5 figuras resumen a partir de los resultados crudos de las
simulaciones (mismo formato que usa extract_model_pilot.py / parallel.py):

    1. Tasa de exito promedio vs. ensayo dentro del bloque (los primeros
       n_trials ensayos de cada bloque, igual que extract_model_pilot.py).
    2. Tasa de aprendizaje promedio vs. ensayo dentro del bloque (la
       derivada de la curva de exito: cuanto cambia la probabilidad de
       exito de un ensayo al siguiente, tal como la define la propuesta
       de titulo -- "la curva de aprendizaje, derivada de la curva de
       probabilidad de exito").
    3. Probabilidad de cambio de accion promedio vs. ensayo dentro del
       bloque.
    4. Tasa de exito promedio POR BLOQUE vs. numero de bloque (progreso a
       lo largo de toda la sesion/simulacion, no dentro de un bloque).
    5. Tasa de disparo del GPe (z-score) alineada al momento de eleccion,
       de -2 a +2 segundos, con la media y el error estandar across
       ensayos (requiere que parallel.py haya guardado
       gpe_rate_values_sim*.npy / gpe_rate_times_sim*.npy /
       choice_times_sim*.npy).

Reutiliza las funciones de carga de datos ya existentes en
analyses/thesis_power_analysis/extract_model_pilot.py (load_trials,
discover_sim_ids, blocks_for_phase) en vez de duplicarlas.

Uso
---
    cd analyses/thesis_figures
    python plot_results.py

Las figuras se guardan como PNG en figures/ (junto a este script) y,
ademas, se intentan mostrar en pantalla si hay una pantalla disponible.

Que ajustar antes de usarlo con otros datos
--------------------------------------------
SOURCES, mas abajo, es una lista de (carpeta, sim_ids) -- igual que
PHASES en extract_model_pilot.py. Por defecto apunta a
simulations/new_task/1_srcSim. Cambiarlo si los datos a graficar estan
en otra carpeta, o agregar mas de una entrada para juntar varias
carpetas/sim_ids en un mismo grafico.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reutiliza las funciones de carga ya existentes en extract_model_pilot.py
# (viven en una carpeta distinta de este repo).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analyses" / "thesis_power_analysis"))

from extract_model_pilot import (  # noqa: E402
    discover_sim_ids,
    load_trials,
    blocks_for_phase,
)
from bootstrap_power import load_trial_matrix  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURACION: de donde salen los datos.
# Lista de (carpeta, sim_ids); sim_ids=None busca todas las simulaciones
# disponibles en esa carpeta (todo archivo output_simN).
# ---------------------------------------------------------------------------
SOURCES: list[tuple[Path, list[int] | None]] = [
    #(REPO_ROOT / "simulations" / "new_task" / "1_srcSim", None),
    #(REPO_ROOT / "stashed_runs" / "new_task" / "first full run new task and block def - 30 sims 60 blocks - 30-09-2026", None),
    (REPO_ROOT / "stashed_runs" / "new_task" / "full run optuna parameters - 30 sims 60 blocks - 07-10-2026", None),
]

# Datos NHP reales (fase "naive"), para superponer en los graficos 1-4 como
# linea roja de comparacion. Mismos archivos que FILES["naive"] en
# run_power_analysis.py. Si no se encuentran, los graficos se muestran
# igual, solo sin la linea NHP (se imprime un aviso).
NHP_SUCCESS_FILE = REPO_ROOT / "analyses" / "thesis_power_analysis" / "behaviorSuccessNaive.xlsx"
NHP_SWITCH_FILE = REPO_ROOT / "analyses" / "thesis_power_analysis" / "behaviorSwitchNaive.xlsx"
NHP_SHEET = "Sheet1"

N_TRIALS = 15          # ensayos por bloque extraidos (debe coincidir con extract_model_pilot.py)
GPE_WINDOW_MS = 1800.0  # +-1.8s alrededor de la eleccion: margen garantizado por
# la estructura del trial es de 1830ms exactos en cada direccion (130ms retraso
# dopamina + 1700ms inter_trial, fijo, independiente de la latencia de decision),
# asi que 1800ms queda dentro de ese limite con margen de seguridad.
GPE_GRID_STEP_MS = 1.0  # debe coincidir con gpe_fr_bin_ms en parallel.py

OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Estilo (ver skill de dataviz): un solo color secuencial (azul), lineas
# finas, grilla horizontal tenue, ejes/leyendas en tinta secundaria/muted,
# sin bordes superiores/derechos.
# ---------------------------------------------------------------------------
LINE_COLOR = "#1c5cab"
BAND_COLOR = "#2a78d6"
BAND_ALPHA = 0.18
NHP_LINE_COLOR = "#b23a3a"
NHP_BAND_COLOR = "#e34948"
GRID_COLOR = "#e1e0d9"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
SURFACE = "#fcfcfb"


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=PRIMARY_INK)
    ax.set_ylabel(ylabel, color=PRIMARY_INK)
    ax.set_title(title, color=PRIMARY_INK, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED_INK)
    ax.spines["bottom"].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _mean_sem(matrix: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Media y error estandar ignorando NaN (bloques mas cortos que
    N_TRIALS, ver extract_model_pilot.py)."""
    mean = np.nanmean(matrix, axis=axis)
    n_valid = np.sum(~np.isnan(matrix), axis=axis)
    sd = np.nanstd(matrix, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = sd / np.sqrt(n_valid)
    return mean, sem


def _mean_sd(matrix: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviacion estandar ignorando NaN (banda mas ancha que
    _mean_sem: reflejan la dispersion entre ensayos, no la incertidumbre
    del promedio)."""
    mean = np.nanmean(matrix, axis=axis)
    sd = np.nanstd(matrix, axis=axis, ddof=1)
    return mean, sd


def _save_and_maybe_show(fig, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> guardado en {out_path}")
    try:
        with warnings.catch_warnings():
            # Sin pantalla disponible (p.ej. corriendo en el cluster con el
            # backend Agg), matplotlib solo emite un UserWarning, no una
            # excepcion -- se silencia porque la figura ya quedo guardada.
            warnings.simplefilter("ignore", UserWarning)
            plt.show(block=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1-3: curvas promedio vs. ensayo dentro del bloque (exito, aprendizaje, cambio)
# ---------------------------------------------------------------------------

def plot_success_vs_trial(success_matrix: np.ndarray, nhp_success_matrix: np.ndarray | None = None) -> None:
    mean, sem = _mean_sem(success_matrix, axis=0)
    trial = np.arange(1, success_matrix.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(trial, mean - sem, mean + sem, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(trial, mean, color=LINE_COLOR, linewidth=1.8, label="Model")

    if nhp_success_matrix is not None:
        nhp_mean, nhp_sem = _mean_sem(nhp_success_matrix, axis=0)
        nhp_trial = np.arange(1, nhp_success_matrix.shape[1] + 1)
        ax.fill_between(nhp_trial, nhp_mean - nhp_sem, nhp_mean + nhp_sem, color=NHP_BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(nhp_trial, nhp_mean, color=NHP_LINE_COLOR, linewidth=1.8, label="NHP")
        ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    _style_axes(ax, "trial number", "success rate", "Avg success rate vs trial")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_and_maybe_show(fig, "1_success_vs_trial.png")


def plot_learning_rate_vs_trial(success_matrix: np.ndarray, nhp_success_matrix: np.ndarray | None = None) -> None:
    # Derivada por bloque (diferencia entre ensayos consecutivos), promediada
    # despues entre bloques -- no se deriva la curva ya promediada, para que
    # el NaN de un bloque corto (ver extract_model_pilot.py) se propague
    # correctamente solo a esa fila en vez de distorsionar los demas puntos.
    row_diff = np.diff(success_matrix, axis=1)
    mean, sem = _mean_sem(row_diff, axis=0)
    # cada punto i representa el cambio entre el ensayo i e i+1 (1-indexado)
    trial_from = np.arange(1, success_matrix.shape[1])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.fill_between(trial_from, mean - sem, mean + sem, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(trial_from, mean, color=LINE_COLOR, linewidth=1.8, label="Model")

    if nhp_success_matrix is not None:
        nhp_row_diff = np.diff(nhp_success_matrix, axis=1)
        nhp_mean, nhp_sem = _mean_sem(nhp_row_diff, axis=0)
        nhp_trial_from = np.arange(1, nhp_success_matrix.shape[1])
        ax.fill_between(nhp_trial_from, nhp_mean - nhp_sem, nhp_mean + nhp_sem, color=NHP_BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(nhp_trial_from, nhp_mean, color=NHP_LINE_COLOR, linewidth=1.8, label="NHP")
        ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    _style_axes(ax, "trial number", "learning slope",
                "Avg learning slope vs trial")
    fig.tight_layout()
    _save_and_maybe_show(fig, "2_learning_rate_vs_trial.png")


def plot_switch_vs_trial(switch_matrix: np.ndarray, nhp_switch_matrix: np.ndarray | None = None) -> None:
    mean, sem = _mean_sem(switch_matrix, axis=0)
    trial = np.arange(1, switch_matrix.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(trial, mean - sem, mean + sem, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(trial, mean, color=LINE_COLOR, linewidth=1.8, label="Model")

    if nhp_switch_matrix is not None:
        nhp_mean, nhp_sem = _mean_sem(nhp_switch_matrix, axis=0)
        nhp_trial = np.arange(1, nhp_switch_matrix.shape[1] + 1)
        ax.fill_between(nhp_trial, nhp_mean - nhp_sem, nhp_mean + nhp_sem, color=NHP_BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(nhp_trial, nhp_mean, color=NHP_LINE_COLOR, linewidth=1.8, label="NHP")
        ax.legend(frameon=False, labelcolor=PRIMARY_INK)

    _style_axes(ax, "trial number", "switch chance", "Avg switch chance vs trial")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_and_maybe_show(fig, "3_switch_vs_trial.png")


# ---------------------------------------------------------------------------
# 4: tasa de exito promedio POR BLOQUE vs. numero de bloque
# ---------------------------------------------------------------------------

def _block_success_by_index(folder: Path, sim_id: int, n_trials: int) -> dict[int, float]:
    """Tasa de exito de cada bloque de una simulacion, indexada por el
    numero de bloque (0-indexado), usando los mismos primeros n_trials
    ensayos que blocks_from_simulation (pero sin pooling, para poder
    promediar 'bloque 5 de todas las simulaciones' mas abajo)."""
    trials = load_trials(folder, sim_id)
    failed_path = folder / f"failed_blocks_sim{sim_id}.npy"
    failed_blocks = np.load(failed_path) if failed_path.exists() else None
    success_all = (trials["decision"] == trials["correct"]).astype(float)

    rates: dict[int, float] = {}
    for block_id, block_trials in trials.groupby("block"):
        if failed_blocks is not None and block_id < len(failed_blocks) and failed_blocks[block_id] == 1:
            continue
        idx = block_trials.index[:n_trials]
        row = success_all.loc[idx].to_numpy()
        if len(row) == 0:
            continue
        rates[int(block_id)] = float(np.mean(row))
    return rates


def plot_success_vs_block_number(sources, n_trials: int) -> None:
    by_block: dict[int, list[float]] = {}
    for folder, sim_ids in sources:
        ids = sim_ids if sim_ids is not None else discover_sim_ids(folder)
        for sim_id in ids:
            for block_id, rate in _block_success_by_index(folder, sim_id, n_trials).items():
                by_block.setdefault(block_id, []).append(rate)

    if not by_block:
        print("  [aviso] no hay datos para el grafico de exito vs. numero de bloque; se omite.")
        return

    block_ids = np.array(sorted(by_block))
    means = np.array([np.mean(by_block[b]) for b in block_ids])
    sems = np.array([
        (np.std(by_block[b], ddof=1) / np.sqrt(len(by_block[b]))) if len(by_block[b]) > 1 else 0.0
        for b in block_ids
    ])
    block_number = block_ids + 1  # 1-indexado para mostrar

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(block_number, means - sems, means + sems, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(block_number, means, color=LINE_COLOR, linewidth=1.8)
    _style_axes(ax, "Block number", "succes rate", "Avg succes rate vs block (model)")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_and_maybe_show(fig, "4_success_vs_block_number.png")


def plot_nhp_success_vs_block_number(nhp_success_matrix: np.ndarray) -> None:
    # Un valor por bloque (no un promedio de repeticiones en cada x, como en
    # la version del modelo), ya que los datos NHP son una unica secuencia
    # cronologica de bloques: se grafica sin banda de error.
    nhp_rates = np.nanmean(nhp_success_matrix, axis=1)
    block_number = np.arange(1, len(nhp_rates) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(block_number, nhp_rates, color=NHP_LINE_COLOR, linewidth=1.0, alpha=0.8)
    _style_axes(ax, "Block number", "succes rate", "Avg succes rate vs block (NHP)")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_and_maybe_show(fig, "4b_nhp_success_vs_block_number.png")


# ---------------------------------------------------------------------------
# 5: tasa de disparo del GPe (z-score) alineada a la eleccion
# ---------------------------------------------------------------------------

def _gpe_trial_traces(sources, window_ms: float, grid_step_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (grid_ms, traces) donde traces tiene forma
    (n_ensayos_totales, len(grid_ms)): la tasa de disparo del GPe (z-score,
    promediada entre las sub-poblaciones de accion) interpolada sobre una
    grilla de tiempo comun relativa al momento de eleccion de cada ensayo.

    El z-score se calcula por SIMULACION (con la media/DE de todo el largo
    de esa simulacion), no por ensayo, ya que cada simulacion tiene su
    propio nivel basal de excitabilidad. Un ensayo cerca del inicio/final
    de una simulacion que no alcance a cubrir +-window_ms completos queda
    con NaN fuera del rango disponible (se ignora al promediar, no se
    descarta el ensayo entero).
    """
    grid_ms = np.arange(-window_ms, window_ms + grid_step_ms, grid_step_ms)
    traces = []

    for folder, sim_ids in sources:
        ids = sim_ids if sim_ids is not None else discover_sim_ids(folder)
        for sim_id in ids:
            rv_path = folder / f"gpe_rate_values_sim{sim_id}.npy"
            rt_path = folder / f"gpe_rate_times_sim{sim_id}.npy"
            ct_path = folder / f"choice_times_sim{sim_id}.npy"
            if not (rv_path.exists() and rt_path.exists() and ct_path.exists()):
                print(f"  [aviso] sim {sim_id}: no se encontraron datos de GPe (gpe_rate_*/choice_times); se omite.")
                continue

            rate_values = np.load(rv_path)   # (n_bins, num_actions)
            rate_times = np.load(rt_path)    # (n_bins,), ms, reloj absoluto de la simulacion
            choice_times = np.load(ct_path)  # (n_ensayos,), ms, mismo reloj

            pooled_rate = rate_values.mean(axis=1)  # promedio entre todas las sub-poblaciones de accion
            sd_r = pooled_rate.std(ddof=1)
            if sd_r == 0:
                continue
            z = (pooled_rate - pooled_rate.mean()) / sd_r

            for choice_time in choice_times:
                rel_t = rate_times - choice_time
                mask = (rel_t >= -window_ms) & (rel_t <= window_ms)
                if mask.sum() < 2:
                    continue
                interp = np.interp(grid_ms, rel_t[mask], z[mask], left=np.nan, right=np.nan)
                traces.append(interp)

    return grid_ms, np.array(traces) if traces else np.empty((0, len(grid_ms)))


def plot_gpe_firing_rate(sources, window_ms: float, grid_step_ms: float) -> None:
    grid_ms, traces = _gpe_trial_traces(sources, window_ms, grid_step_ms)
    if len(traces) == 0:
        print("  [aviso] no hay datos de GPe disponibles (corre parallel.py con la version que "
              "guarda gpe_rate_values_sim*/gpe_rate_times_sim*/choice_times_sim*); se omite este grafico.")
        return

    mean, sd = _mean_sd(traces, axis=0)
    time_s = grid_ms / 1000.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(0, color=MUTED_INK, linewidth=1.0, linestyle="--")
    ax.fill_between(time_s, mean - sd, mean + sd, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(time_s, mean, color=LINE_COLOR, linewidth=1.8)
    _style_axes(ax, "time (seconds)", "GPe FR (z)", f"GPe firing rate (z) around choice selection\n(n={len(traces)} trials)")
    ax.set_xlim(-window_ms / 1000.0, window_ms / 1000.0)
    fig.tight_layout()
    _save_and_maybe_show(fig, "5_gpe_firing_rate.png")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                         help=f"ensayos por bloque a usar para los graficos 1-4 (default: {N_TRIALS})")
    parser.add_argument("--window-ms", type=float, default=GPE_WINDOW_MS,
                         help=f"ventana +- alrededor de la eleccion para el grafico de GPe, en ms (default: {GPE_WINDOW_MS})")
    parser.add_argument("--grid-step-ms", type=float, default=GPE_GRID_STEP_MS,
                         help=f"paso de la grilla temporal para el grafico de GPe, en ms (default: {GPE_GRID_STEP_MS})")
    args = parser.parse_args()

    print("Cargando bloques (primeros", args.n_trials, "ensayos por bloque)...")
    success_matrix, switch_matrix = blocks_for_phase(SOURCES, args.n_trials)
    print(f"  -> {success_matrix.shape[0]} bloques (exito), {switch_matrix.shape[0]} bloques (cambio)")

    print("Cargando datos NHP (fase naive) para comparacion...")
    try:
        nhp_success_matrix = load_trial_matrix(NHP_SUCCESS_FILE, sheet_name=NHP_SHEET)
        print(f"  -> {nhp_success_matrix.shape[0]} bloques NHP (exito)")
    except Exception as e:
        print(f"  [aviso] No se pudo cargar {NHP_SUCCESS_FILE} ({e}); los graficos se mostraran sin la linea NHP.")
        nhp_success_matrix = None
    try:
        nhp_switch_matrix = load_trial_matrix(NHP_SWITCH_FILE, sheet_name=NHP_SHEET)
        print(f"  -> {nhp_switch_matrix.shape[0]} bloques NHP (cambio)")
    except Exception as e:
        print(f"  [aviso] No se pudo cargar {NHP_SWITCH_FILE} ({e}); los graficos se mostraran sin la linea NHP.")
        nhp_switch_matrix = None

    print("\nGraficos 1-3: curvas vs. ensayo dentro del bloque")
    plot_success_vs_trial(success_matrix, nhp_success_matrix)
    plot_learning_rate_vs_trial(success_matrix, nhp_success_matrix)
    plot_switch_vs_trial(switch_matrix, nhp_switch_matrix)

    print("\nGrafico 4: exito vs. numero de bloque (modelo y NHP por separado)")
    plot_success_vs_block_number(SOURCES, args.n_trials)
    if nhp_success_matrix is not None:
        plot_nhp_success_vs_block_number(nhp_success_matrix)

    print("\nGrafico 5: tasa de disparo del GPe vs. tiempo")
    plot_gpe_firing_rate(SOURCES, args.window_ms, args.grid_step_ms)

    print(f"\nListo. Figuras guardadas en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
