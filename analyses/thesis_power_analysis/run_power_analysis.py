"""
Script principal: ejecuta el analisis de potencia bootstrap para las 9 celdas
(3 metricas x 3 fases) y produce una tabla resumen.

INSTRUCCIONES DE USO CON DATOS REALES:
Reemplazar el diccionario FILES mas abajo con las rutas/hojas reales de los
archivos de datos NHP y de las simulaciones piloto del modelo. Cada entrada
debe apuntar a una matriz de bloques x ensayos (0/1) para exito y para
cambio, tanto para NHP como para el modelo, en cada una de las 3 fases.

DATOS AUN NO DISPONIBLES:
Si a un archivo/hoja le falta (p.ej. porque aun no se ha corrido esa
simulacion, o el archivo/hoja de Excel todavia no existe), este script no
se interrumpe: registra el problema, imprime un aviso, y en la tabla final
deja "-" en cada celda que no se pudo calcular por esa razon (en vez de
detener todo el analisis). Si de una fase falta solo un lado (p.ej. NHP
esta pero el modelo aun no), igual se muestran las estadisticas del lado
que si esta disponible.
"""

import numpy as np
import pandas as pd
from bootstrap_power import (
    load_trial_matrix,
    METRIC_FUNCS,
    find_min_n2_bootstrap,
)

# ---------------------------------------------------------------------------
# Configuracion: margenes de equivalencia estandarizados (d) por fase.
# Se aplica el mismo d a las 3 metricas dentro de una fase, salvo que se
# justifique lo contrario (ver metodologia).
# ---------------------------------------------------------------------------
D_BY_PHASE = {"naive": 0.20, "pcp": 0.20, "dbs": 0.30}

# Correccion de Bonferroni: 3 metricas evaluadas por fase (familia de pruebas)
ALPHA = 0.05 / 3

TARGET_POWER = 0.80
N_SIMS = 10000       # subir a 5000-10000 para el resultado final del informe
SEED = 54321


# ---------------------------------------------------------------------------
# Rutas de archivos (AJUSTAR a la estructura real de los datos)
# ---------------------------------------------------------------------------
FILES = {
    "naive": {
        "nhp_success": ("behaviorSuccessNaive.xlsx", "Sheet1"),
        "nhp_switch":  ("behaviorSwitchNaive.xlsx", "Sheet1"),
        "model_success": ("model_pilot.xlsx", "demo_success"),
        "model_switch":  ("model_pilot.xlsx", "demo_switch"),
    },
    "pcp": {
        "nhp_success": ("behaviorSuccessPcp.xlsx", "Sheet1"),
        "nhp_switch":  ("behaviorSwitchPcp.xlsx", "Sheet1"),
        "model_success": ("model_pilot.xlsx", "pcp_success"),
        "model_switch":  ("model_pilot.xlsx", "pcp_switch"),
    },
    "dbs": {
        "nhp_success": ("behaviorSuccess13Hz.xlsx", "Sheet1"),
        "nhp_switch":  ("behaviorSwitch13Hz.xlsx", "Sheet1"),
        "model_success": ("model_pilot.xlsx", "dbs_success"),
        "model_switch":  ("model_pilot.xlsx", "dbs_switch"),
    },
}


NA = "-"  # marcador para toda celda que no se pudo calcular por datos faltantes


def load_phase_data(phase_files):
    """
    Carga y limpia las 4 matrices (NHP/modelo x exito/cambio) de una fase.

    Si un archivo o una hoja no existe todavia (p.ej. la simulacion de esa
    fase aun no se corrio, o el Excel de NHP aun no fue entregado), no se
    interrumpe la carga de las demas fuentes: esa entrada queda en None y se
    imprime un aviso, para que compute_metric_arrays/run_full_analysis
    puedan marcar con "-" solo lo que realmente falta.
    """
    out = {}
    for key, (path, sheet) in phase_files.items():
        print(f"Cargando {key}: {path} [{sheet}]")
        try:
            out[key] = load_trial_matrix(path, sheet_name=sheet)
        except Exception as e:
            print(f"  [aviso] No se pudo cargar {path} [{sheet}] ({e}); "
                  f"se usara '-' donde dependa de esta fuente.")
            out[key] = None
    return out


def compute_metric_arrays(matrices):
    """
    A partir de las 4 matrices crudas de una fase (algunas posiblemente
    None, ver load_phase_data), calcula los 3 pares (NHP, modelo) de
    arreglos por-bloque para cada metrica. Un par queda como (None, ...) o
    (..., None) si a la matriz de origen le falto alguno de los dos lados.
    """
    def safe_metric(func, matrix):
        return func(matrix) if matrix is not None else None

    result = {}
    result["success"] = (
        safe_metric(METRIC_FUNCS["success"], matrices["nhp_success"]),
        safe_metric(METRIC_FUNCS["success"], matrices["model_success"]),
    )
    result["switch"] = (
        safe_metric(METRIC_FUNCS["switch"], matrices["nhp_switch"]),
        safe_metric(METRIC_FUNCS["switch"], matrices["model_switch"]),
    )
    result["learning_rate"] = (
        safe_metric(METRIC_FUNCS["learning_rate"], matrices["nhp_success"]),
        safe_metric(METRIC_FUNCS["learning_rate"], matrices["model_success"]),
    )
    return result


def run_full_analysis(files_dict=FILES, assume_null_true=False):
    rows = []
    for phase, phase_files in files_dict.items():
        print(f"\n=== Fase: {phase} ===")
        matrices = load_phase_data(phase_files)
        metric_arrays = compute_metric_arrays(matrices)
        d = D_BY_PHASE[phase]

        for metric_name, (x1, x2_pool) in metric_arrays.items():
            row = {
                "fase": phase,
                "metrica": metric_name,
                "media_NHP": NA,
                "SD_NHP": NA,
                "media_modelo_piloto": NA,
                "SD_modelo_piloto": NA,
                "d_SESOI": d,
                "margen_equivalencia": NA,
                "n1_NHP": NA,
                "n2_minimo": NA,
                "potencia_lograda": NA,
            }

            # Estadisticas del lado NHP (posibles aunque falte el modelo)
            margin = None
            if x1 is not None:
                n1 = len(x1)
                sd_nhp = np.std(x1, ddof=1)
                mean_nhp = np.mean(x1)
                margin = d * sd_nhp
                row.update({
                    "media_NHP": mean_nhp, "SD_NHP": sd_nhp,
                    "n1_NHP": n1, "margen_equivalencia": margin,
                })

            # Estadisticas del lado modelo (posibles aunque falte el NHP)
            if x2_pool is not None:
                row.update({
                    "media_modelo_piloto": np.mean(x2_pool),
                    "SD_modelo_piloto": np.std(x2_pool, ddof=1),
                })

            if x1 is None or x2_pool is None:
                faltante = []
                if x1 is None:
                    faltante.append("NHP")
                if x2_pool is None:
                    faltante.append("modelo")
                print(f"  -> Metrica: {metric_name}: faltan datos de "
                      f"{' y '.join(faltante)}; potencia no calculable, "
                      f"se deja '-'.")
                rows.append(row)
                continue

            print(f"  -> Metrica: {metric_name} "
                  f"(media NHP={row['media_NHP']:.4f}, SD NHP={row['SD_NHP']:.4f}, "
                  f"n1={row['n1_NHP']}, margen={margin:.4f})")

            try:
                n2, power = find_min_n2_bootstrap(
                    x1, x2_pool, n1=row["n1_NHP"], margin=margin, alpha=ALPHA,
                    target_power=TARGET_POWER, n_sims=N_SIMS,
                    assume_null_true=assume_null_true, seed=SEED,
                )
                row["n2_minimo"] = n2
                row["potencia_lograda"] = power
            except Exception as e:
                print(f"  [aviso] Fallo el calculo de potencia para "
                      f"{phase}/{metric_name} ({e}); se deja '-'.")

            rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results_df = run_full_analysis()
    print("\n\n=== TABLA RESUMEN ===")
    print(results_df.to_string(index=False))
    results_df.to_csv("power_analysis_results.csv", index=False)
    print("\nResultados guardados en power_analysis_results.csv")
