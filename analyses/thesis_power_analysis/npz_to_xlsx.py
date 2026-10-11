"""
Genera 4 archivos .xlsx (exito/cambio x calibracion/validacion) para la fase
"naive", en el mismo formato que load_trial_matrix() espera (sin header, sin
indice), a partir de los datos NHP completos y la MISMA particion
calibracion/validacion (semilla y fraccion) que usa hpo_search.py.

Por que no se lee todo directamente de un .npz existente
----------------------------------------------------------
hpo_search.py ya guarda un .npz con los bloques de VALIDACION completos
(naive_validation_blocks.npz: success_validation/switch_validation), pero
el de CALIBRACION (naive_calibration_split.npz) solo guarda las
estadisticas resumen (media/DE) que necesita su funcion objetivo -- nunca
la matriz de bloques completa, porque Optuna no la necesita. Para poder
correr run_power_analysis.py sobre los bloques de calibracion (no solo sus
estadisticas), hace falta la matriz completa.

En vez de modificar el cache de hpo_search.py (lo que forzaria un
re-split y podria interferir con un estudio de Optuna en curso), este
script RE-DERIVA la particion completa (calibracion y validacion, exito y
cambio) directamente desde los archivos NHP originales, usando la MISMA
semilla/fraccion que hpo_search.py (CALIBRATION_FRACTION=0.70,
CALIBRATION_SEED=20240101, ver _split_indices alli). Como el split es
determinista, esto reproduce EXACTAMENTE la misma particion -- y este
script lo verifica comparando los bloques de validacion recalculados
contra los ya guardados en naive_validation_blocks.npz antes de escribir
nada, como chequeo de que la reproduccion es correcta.

Salida
------
    behaviorSuccessNaiveCalibration.xlsx  (hoja "Sheet1")
    behaviorSwitchNaiveCalibration.xlsx   (hoja "Sheet1")
    behaviorSuccessNaiveValidation.xlsx   (hoja "Sheet1")
    behaviorSwitchNaiveValidation.xlsx    (hoja "Sheet1")

Uso
---
    python npz_to_xlsx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSES_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSES_DIR.parents[1]

from bootstrap_power import load_trial_matrix  # noqa: E402

# Deben coincidir EXACTAMENTE con hpo_search.py (ver CALIBRATION_FRACTION/
# CALIBRATION_SEED/_split_indices alli). Se copia la logica de split (en
# vez de importar la version privada _split_indices de hpo_search.py) para
# no depender de un simbolo "privado" de otro modulo, pero debe mantenerse
# identica si esos valores cambian alli.
CALIBRATION_FRACTION = 0.70
CALIBRATION_SEED = 20240101


def _split_indices(n: int, calib_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_calib = int(round(n * calib_frac))
    return idx[:n_calib], idx[n_calib:]


def _write_xlsx(matrix: np.ndarray, path: Path) -> None:
    pd.DataFrame(matrix).to_excel(path, sheet_name="Sheet1", header=False, index=False)
    print(f"  -> {path} ({matrix.shape[0]} bloques)")


def main() -> None:
    success_full = load_trial_matrix(ANALYSES_DIR / "behaviorSuccessNaive.xlsx", sheet_name="Sheet1")
    switch_full = load_trial_matrix(ANALYSES_DIR / "behaviorSwitchNaive.xlsx", sheet_name="Sheet1")

    success_calib_idx, success_valid_idx = _split_indices(len(success_full), CALIBRATION_FRACTION, CALIBRATION_SEED)
    switch_calib_idx, switch_valid_idx = _split_indices(len(switch_full), CALIBRATION_FRACTION, CALIBRATION_SEED + 1)

    success_calib = success_full[success_calib_idx]
    success_valid = success_full[success_valid_idx]
    switch_calib = switch_full[switch_calib_idx]
    switch_valid = switch_full[switch_valid_idx]

    # --- Verificacion: la validacion recalculada debe coincidir EXACTO con
    # la ya guardada por hpo_search.py, antes de confiar en que el split
    # (y por lo tanto el lado de calibracion, que nunca se guardo
    # explicitamente) se reprodujo correctamente.
    cache_path = ANALYSES_DIR / "naive_validation_blocks.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        success_match = np.array_equal(cached["success_validation"], success_valid)
        switch_match = np.array_equal(cached["switch_validation"], switch_valid)
        if success_match and switch_match:
            print(f"[verificacion] OK: la validacion recalculada coincide exacto con {cache_path}")
        else:
            print(f"[ADVERTENCIA] La validacion recalculada NO coincide con {cache_path} "
                  f"(success_match={success_match}, switch_match={switch_match}). "
                  f"Esto significa que CALIBRATION_FRACTION/CALIBRATION_SEED aqui no coinciden "
                  f"con hpo_search.py, o que el .npz cacheado es de una corrida anterior con "
                  f"otra semilla. No continuar sin resolver esto -- los archivos generados abajo "
                  f"no corresponderian al split real que uso Optuna.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[aviso] No se encontro {cache_path} (hpo_search.py aun no corrio la calibracion); "
              f"no se pudo verificar el split, pero se continua igual ya que es determinista.")

    print("\nEscribiendo archivos:")
    _write_xlsx(success_calib, ANALYSES_DIR / "behaviorSuccessNaiveCalibration.xlsx")
    _write_xlsx(switch_calib, ANALYSES_DIR / "behaviorSwitchNaiveCalibration.xlsx")
    _write_xlsx(success_valid, ANALYSES_DIR / "behaviorSuccessNaiveValidation.xlsx")
    _write_xlsx(switch_valid, ANALYSES_DIR / "behaviorSwitchNaiveValidation.xlsx")
    print("\nListo.")


if __name__ == "__main__":
    main()
