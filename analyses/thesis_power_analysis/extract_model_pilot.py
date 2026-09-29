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
    3. Para cada bloque restante, toma los ULTIMOS `n_trials` ensayos
       (todo bloque no-fallido tiene garantizados al menos 15 ensayos,
       porque el criterio de exito de parallel.py exige 15 aciertos
       CONSECUTIVOS antes de terminar el bloque), para igualar la
       convencion usada al extraer los datos NHP (alli tambien se usan
       los ultimos ensayos del bloque). Esto NO reconcilia la definicion
       de "bloque" en si entre modelo y NHP (ver caveat en
       blocks_from_simulation); solo iguala que ensayo se toma como
       primero/ultimo dentro de cada bloque.
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
    Devuelve (success_rows, switch_rows) para una simulacion: dos arreglos
    de forma (n_bloques_no_fallidos, n_trials), con NaN donde un bloque
    (que no deberia pasar, salvo casos raros) tuviera menos de n_trials
    ensayos.
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

        # Se toman los ULTIMOS n_trials ensayos del bloque (no los primeros),
        # para igualar la convencion usada al extraer los datos NHP.
        # NOTA: esto no reconcilia la definicion de "bloque" en si -en el
        # modelo un bloque termina tras 15 aciertos consecutivos (max. 30
        # ensayos), mientras que en los datos NHP termina con 12-15 aciertos
        # dentro de los ultimos 25 ensayos, con el umbral elegido al azar por
        # bloque-, solo iguala que ambos lados aporten los ensayos "finales"
        # del bloque en vez de los "iniciales". Pendiente: unificar la
        # definicion de bloque en el codigo de simulacion y re-extraer.
        idx = block_trials.index[-n_trials:]
        row_success = success_all.loc[idx].to_numpy()
        row_switch = switch_all.loc[idx].to_numpy()

        if len(idx) < n_trials:
            # No deberia pasar para un bloque no-fallido (todo bloque exitoso
            # tiene >=15 ensayos, ya que el criterio de exito exige 15
            # aciertos consecutivos), pero se protege igual: se rellena con
            # NaN al INICIO (para conservar el ultimo ensayo real en la
            # ultima columna) y load_trial_matrix descartara la fila.
            pad = n_trials - len(idx)
            row_success = np.concatenate([np.full(pad, np.nan), row_success])
            row_switch = np.concatenate([np.full(pad, np.nan), row_switch])

        success_rows.append(row_success)
        switch_rows.append(row_switch)

    if not success_rows:
        return (np.empty((0, n_trials)), np.empty((0, n_trials)))

    return np.vstack(success_rows), np.vstack(switch_rows)


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
            if len(success_rows):
                all_success.append(success_rows)
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
