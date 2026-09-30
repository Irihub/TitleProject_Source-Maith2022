"""
Construye model_pilot.xlsx (los datos "piloto" del modelo que espera
bootstrap_power.py / run_power_analysis.py) a partir de resultados crudos de
simulaciones (output_simN, selection_simN.npy, failed_blocks_simN.npy,
producidos por parallel.py).

Este script IGNORA por completo los datos NHP; solo prepara el lado
"modelo" (model_pilot.xlsx). Los datos NHP reales (nhp_data.xlsx) los
maneja el usuario por separado.

-------------------------------------------------------------------------
Por que hace falta este paso
-------------------------------------------------------------------------
bootstrap_power.load_trial_matrix() espera, por hoja de Excel, una matriz
de bloques x ensayos (ancho fijo, `n_trials` columnas) en la que cada fila
es un bloque y cada columna un ensayo dentro del bloque, con NaN si el
bloque no completo esa cantidad de ensayos (y esas filas se descartan por
completo). Los datos crudos de las simulaciones, en cambio, vienen en
formato "una fila = un ensayo" (output_simN) mas un archivo de bloques
fallidos (failed_blocks_simN.npy). Este script hace la conversion:

    1. Para cada simulacion, lee output_simN (con las columnas
       trial/correct/decision) y agrupa los ensayos en bloques: un bloque
       nuevo empieza cada vez que cambia la posicion recompensada
       ("correct"), tal como hace parallel.py en su propio bucle `for
       block in range(num_blocks)`.
    2. Descarta enteramente los bloques marcados como fallidos en
       failed_blocks_simN.npy (equivalente a los bloques incompletos que
       load_trial_matrix descarta para los datos NHP).
    3. Para cada bloque restante, toma los PRIMEROS `n_trials` ensayos,
       para igualar la convencion usada al extraer los datos NHP (los 15
       ensayos reportados alli son los PRIMEROS de cada bloque, no los
       ultimos). Esto NO reconcilia la definicion de "bloque" en si entre
       modelo y NHP (ver caveat en blocks_from_simulation); solo iguala
       que ensayo se toma como primero dentro de cada bloque.
    4. Calcula, para cada ensayo, dos metricas binarias:
         - exito:  1 si decision == correct, 0 si no.
         - cambio: 1 si la decision de este ensayo es distinta de la
           decision del ensayo INMEDIATAMENTE anterior en la misma
           simulacion (cruzando el limite entre bloques, igual que en una
           sesion real continua). El primer ensayo de cada simulacion no
           tiene ensayo anterior y queda como NaN (esa fila se descartara
           downstream, igual que un bloque incompleto).
    5. Junta los bloques de todas las simulaciones/carpetas asignadas a
       una misma "fase" en una unica matriz, y la guarda en una hoja
       "{fase}_success"/"{fase}_switch" de model_pilot.xlsx, exactamente
       en el formato que produce make_test_data.py (sin header, sin
       indice) y que load_trial_matrix sabe leer.

-------------------------------------------------------------------------
Que hay que ajustar antes de usarlo con datos reales
-------------------------------------------------------------------------
El diccionario PHASES, mas abajo, es solo un ejemplo (usa los 5 pilotos de
014a_new_dopamine_cluster que ya estan en este repo, todos bajo una unica
fase "demo"). Reemplazarlo por el mapeo real fase -> carpeta(s) de
simulacion de la tesis (p.ej. distintas carpetas en simulations/ para
"naive"/"pcp"/"dbs", o distintos valores de sim_id dentro de una misma
carpeta). Una fase puede juntar bloques de varias carpetas/sim_ids a la
vez, ya que `find_sim_ids` acepta una lista de (carpeta, [ids]).

Uso
---
    python extract_model_pilot.py
    python extract_model_pilot.py --out model_pilot.xlsx --n-trials 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = [
    "trial",
    "start",
    "dop_input",
    "end",
    "correct",
    "decision",
    "dopamine",
    "time",
]

# ---------------------------------------------------------------------------
# CONFIGURACION: mapear cada fase a una o mas carpetas de simulacion.
#
# Cada entrada de PHASES es una lista de (carpeta, sim_ids). `sim_ids` puede
# ser None para usar todas las simulaciones encontradas en esa carpeta
# (busca todo archivo output_simN).
#
# EJEMPLO (por defecto): usa los 5 pilotos ya presentes en
# simulations/014a_new_dopamine_cluster/1_srcSim como si fueran una unica
# fase "demo", solo para validar que el pipeline corre de punta a punta.
# REEMPLAZAR con las carpetas/fases reales de la tesis.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

PHASES: dict[str, list[tuple[Path, list[int] | None]]] = {
    "demo": [
        (REPO_ROOT / "simulations" / "014a_new_dopamine_cluster" / "1_srcSim", None),
    ],
    # "naive": [(REPO_ROOT / "simulations" / "<carpeta_naive>" / "1_srcSim", None)],
    # "pcp":   [(REPO_ROOT / "simulations" / "<carpeta_pcp>" / "1_srcSim", None)],
    # "dbs":   [(REPO_ROOT / "simulations" / "<carpeta_dbs>" / "1_srcSim", None)],
}


def discover_sim_ids(folder: Path) -> list[int]:
    """Encuentra todo sim id N para el cual existe output_simN en `folder`."""
    ids = []
    for f in folder.glob("output_sim*"):
        suffix = f.name[len("output_sim"):]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def load_trials(folder: Path, sim_id: int) -> pd.DataFrame:
    """Carga output_simN y le agrega una columna 'block' (0-indexed, una
    corrida por cada cambio de 'correct'), igual que en show_results.py."""
    output_path = folder / f"output_sim{sim_id}"
    trials = pd.read_csv(
        output_path, sep="\t", comment="#", header=None, names=OUTPUT_COLUMNS,
        usecols=range(len(OUTPUT_COLUMNS)),
    )
    block_id = (trials["correct"] != trials["correct"].shift()).cumsum() - 1
    trials = trials.assign(block=block_id)
    return trials


def blocks_from_simulation(folder: Path, sim_id: int, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (success_rows, switch_rows) para una simulacion: dos arreglos,
    de forma (n_bloques_utilizables, n_trials) cada uno (no necesariamente
    con el mismo numero de filas entre si, ver abajo).

    Bajo block_definition="legacy" todo bloque no-fallido tiene garantizados
    >=15 ensayos (el criterio de exito exige 15 aciertos CONSECUTIVOS). Bajo
    "nhp_like", en cambio, un bloque puede terminar con TAN POCO como
    nhp_threshold_min ensayos (si el umbral de ese bloque se alcanza antes
    de llegar a n_trials) -- esto ya NO es un caso raro, es esperable. Esos
    bloques cortos se rellenan con NaN al inicio de la fila y luego se
    descartan por completo (independientemente entre exito y cambio, ya
    que un bloque puede quedar incompleto en una metrica y no en la otra:
    el relleno afecta ambas por igual, pero el NaN del primer ensayo de
    cada simulacion en 'cambio' solo afecta esa metrica). Por eso
    success_rows y switch_rows pueden terminar con distinto numero de
    filas -- igual que ya asume blocks_for_phase() al juntarlas por
    separado.
    """
    trials = load_trials(folder, sim_id)

    failed_path = folder / f"failed_blocks_sim{sim_id}.npy"
    failed_blocks = np.load(failed_path) if failed_path.exists() else None

    # "cambio de accion": decision distinta a la del ensayo inmediatamente
    # anterior de la MISMA simulacion (cruza limites de bloque). El primer
    # ensayo de la simulacion no tiene anterior -> NaN.
    switch_all = (trials["decision"] != trials["decision"].shift()).astype(float)
    switch_all.iloc[0] = np.nan
    success_all = (trials["decision"] == trials["correct"]).astype(float)

    success_rows = []
    switch_rows = []
    for block_id, block_trials in trials.groupby("block"):
        if failed_blocks is not None and block_id < len(failed_blocks) and failed_blocks[block_id] == 1:
            continue  # bloque incompleto/fallido: se descarta por completo

        # Se toman los PRIMEROS n_trials ensayos del bloque, para igualar la
        # convencion usada al extraer los datos NHP (los 15 ensayos
        # reportados alli son los primeros de cada bloque, no los ultimos).
        # NOTA: esto no reconcilia la definicion de "bloque" en si -en el
        # modelo un bloque termina tras 15 aciertos consecutivos (max. 30
        # ensayos) o, bajo "nhp_like", tras 12-15 aciertos dentro de los
        # ultimos 25 ensayos (umbral elegido al azar por bloque)-, solo
        # asegura que ambos lados tomen los ensayos "iniciales" del bloque.
        idx = block_trials.index[:n_trials]
        row_success = success_all.loc[idx].to_numpy()
        row_switch = switch_all.loc[idx].to_numpy()

        if len(idx) < n_trials:
            # Bloque mas corto que n_trials (esperable bajo "nhp_like", ver
            # docstring de arriba): se rellena con NaN al FINAL (para
            # conservar el primer ensayo real en la primera columna); la
            # fila se descarta mas abajo.
            pad = n_trials - len(idx)
            row_success = np.concatenate([row_success, np.full(pad, np.nan)])
            row_switch = np.concatenate([row_switch, np.full(pad, np.nan)])

        success_rows.append(row_success)
        switch_rows.append(row_switch)

    if not success_rows:
        return (np.empty((0, n_trials)), np.empty((0, n_trials)))

    success_matrix = np.vstack(success_rows)
    switch_matrix = np.vstack(switch_rows)

    # Descarta filas con NaN, independientemente por metrica (igual que
    # bootstrap_power.load_trial_matrix hace para el lado NHP). Antes esto
    # solo ocurria "gratis" cuando estos datos pasaban por un archivo Excel
    # y se releian con load_trial_matrix (p.ej. en build_model_pilot); un
    # consumidor que use esta funcion directamente (p.ej. hpo_search.py)
    # necesita este filtro aqui mismo, o una sola fila con NaN contamina
    # silenciosamente cualquier .mean() posterior sobre toda la matriz.
    success_matrix = success_matrix[~np.isnan(success_matrix).any(axis=1)]
    switch_matrix = switch_matrix[~np.isnan(switch_matrix).any(axis=1)]

    return success_matrix, switch_matrix


def blocks_for_phase(sources: list[tuple[Path, list[int] | None]], n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Junta los bloques de todas las (carpeta, sim_ids) de una fase."""
    all_success = []
    all_switch = []
    for folder, sim_ids in sources:
        ids = sim_ids if sim_ids is not None else discover_sim_ids(folder)
        if not ids:
            print(f"  [aviso] No se encontraron output_sim* en {folder}")
            continue
        for sim_id in ids:
            success_rows, switch_rows = blocks_from_simulation(folder, sim_id, n_trials)
            # Checked independently: success_rows/switch_rows can now have
            # different lengths (see blocks_from_simulation's docstring).
            if len(success_rows):
                all_success.append(success_rows)
            if len(switch_rows):
                all_switch.append(switch_rows)

    if not all_success:
        return (np.empty((0, n_trials)), np.empty((0, n_trials)))

    return np.vstack(all_success), np.vstack(all_switch)


def build_model_pilot(phases: dict[str, list[tuple[Path, list[int] | None]]],
                       out_path: str, n_trials: int = 15) -> None:
    with pd.ExcelWriter(out_path) as writer:
        for phase, sources in phases.items():
            print(f"Fase '{phase}':")
            success, switch = blocks_for_phase(sources, n_trials)
            print(f"  -> {success.shape[0]} bloques utilizables (de {n_trials} ensayos c/u)")

            pd.DataFrame(success).to_excel(writer, sheet_name=f"{phase}_success", header=False, index=False)
            pd.DataFrame(switch).to_excel(writer, sheet_name=f"{phase}_switch", header=False, index=False)

    print(f"\nArchivo generado: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default="model_pilot.xlsx",
                         help="ruta de salida del archivo .xlsx (default: model_pilot.xlsx)")
    parser.add_argument("--n-trials", type=int, default=15,
                         help="numero fijo de ensayos por bloque a extraer (default: 15)")
    args = parser.parse_args()

    build_model_pilot(PHASES, args.out, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
