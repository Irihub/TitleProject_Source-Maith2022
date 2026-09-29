"""
Genera archivos Excel sinteticos con la MISMA estructura descrita por el
usuario (bloques x 15 ensayos, 0/1, con algunos bloques incompletos) para
validar que el pipeline de analisis de potencia funciona de extremo a
extremo antes de usarlo con los datos reales.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def make_success_matrix(n_blocks, n_trials=15, p_start=0.3, p_end=0.9,
                         incomplete_frac=0.0):
    """Simula una curva de aprendizaje: la probabilidad de exito sube desde
    p_start hasta p_end a lo largo de los 15 ensayos, con ruido bloque a
    bloque, y opcionalmente deja una fraccion de bloques incompletos
    (con NaN en las ultimas columnas), replicando el ejemplo dado por el
    usuario."""
    trial_probs = np.linspace(p_start, p_end, n_trials)
    data = np.zeros((n_blocks, n_trials))
    for b in range(n_blocks):
        block_shift = rng.normal(0, 0.05)
        probs = np.clip(trial_probs + block_shift, 0.01, 0.99)
        data[b] = rng.binomial(1, probs)

    df = pd.DataFrame(data)

    n_incomplete = int(n_blocks * incomplete_frac)
    if n_incomplete > 0:
        idx = rng.choice(n_blocks, size=n_incomplete, replace=False)
        for i in idx:
            cut = rng.integers(2, n_trials - 1)
            df.iloc[i, cut:] = np.nan
    return df


def make_switch_matrix(n_blocks, n_trials=15, p=0.2, incomplete_frac=0.0):
    data = rng.binomial(1, p, size=(n_blocks, n_trials)).astype(float)
    df = pd.DataFrame(data)
    n_incomplete = int(n_blocks * incomplete_frac)
    if n_incomplete > 0:
        idx = rng.choice(n_blocks, size=n_incomplete, replace=False)
        for i in idx:
            cut = rng.integers(2, n_trials - 1)
            df.iloc[i, cut:] = np.nan
    return df


# --- NHP data (real experiment) ---
nhp_specs = {
    "naive": dict(n_blocks=1673, p_start=0.3, p_end=0.92, switch_p=0.15),
    "pcp":   dict(n_blocks=770,  p_start=0.25, p_end=0.72, switch_p=0.275),
    "dbs":   dict(n_blocks=227,  p_start=0.3, p_end=0.80, switch_p=0.21),
}

with pd.ExcelWriter("nhp_data.xlsx") as writer:
    for phase, spec in nhp_specs.items():
        succ = make_success_matrix(spec["n_blocks"], p_start=spec["p_start"],
                                    p_end=spec["p_end"], incomplete_frac=0.005)
        sw = make_switch_matrix(spec["n_blocks"], p=spec["switch_p"],
                                 incomplete_frac=0.005)
        succ.to_excel(writer, sheet_name=f"{phase}_success", header=False, index=False)
        sw.to_excel(writer, sheet_name=f"{phase}_switch", header=False, index=False)

# --- Model pilot data (smaller pilot batch, slightly different to be realistic) ---
model_specs = {
    "naive": dict(n_blocks=120, p_start=0.32, p_end=0.90, switch_p=0.17),
    "pcp":   dict(n_blocks=120, p_start=0.28, p_end=0.70, switch_p=0.29),
    "dbs":   dict(n_blocks=80,  p_start=0.30, p_end=0.78, switch_p=0.23),
}

with pd.ExcelWriter("model_pilot.xlsx") as writer:
    for phase, spec in model_specs.items():
        succ = make_success_matrix(spec["n_blocks"], p_start=spec["p_start"],
                                    p_end=spec["p_end"])
        sw = make_switch_matrix(spec["n_blocks"], p=spec["switch_p"])
        succ.to_excel(writer, sheet_name=f"{phase}_success", header=False, index=False)
        sw.to_excel(writer, sheet_name=f"{phase}_switch", header=False, index=False)

print("Archivos sinteticos de prueba generados: nhp_data.xlsx, model_pilot.xlsx")
