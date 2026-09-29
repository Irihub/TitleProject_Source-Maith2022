"""
Analisis de potencia estadistica a priori para las pruebas de equivalencia (TOST)
entre el modelo computacional y los datos de los NHP, mediante bootstrap no
parametrico.

Estructura esperada de los datos crudos (por fase y por tipo de dato):
    - Cada fila es un bloque.
    - Cada columna es un ensayo (trial) dentro del bloque (hasta 15).
    - Un bloque incompleto tiene celdas vacias (NaN al leerlo con pandas) y
      debe ser descartado en su totalidad antes de calcular cualquier metrica.
    - Para la hoja de "exito": 1 = ensayo exitoso, 0 = fallo.
    - Para la hoja de "cambio": 1 = hubo cambio de accion, 0 = no hubo cambio.

Autor: (adaptar a nombre del estudiante)
"""

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ttost_ind


# ---------------------------------------------------------------------------
# 1. Carga y limpieza de datos
# ---------------------------------------------------------------------------

def load_trial_matrix(source, sheet_name=None):
    """
    Carga una matriz de bloques x ensayos desde un Excel/CSV o un DataFrame ya
    en memoria, y descarta por completo cualquier bloque (fila) incompleto,
    es decir, cualquier fila que contenga al menos un valor faltante (NaN).

    Parameters
    ----------
    source : str | pandas.DataFrame
        Ruta a un archivo .xlsx/.csv, o un DataFrame ya cargado.
    sheet_name : str, optional
        Nombre de la hoja, si `source` es la ruta a un archivo .xlsx con
        multiples hojas (p.ej. una hoja por fase).

    Returns
    -------
    numpy.ndarray
        Matriz de forma (n_bloques_completos, n_ensayos), dtype float.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif str(source).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(source, sheet_name=sheet_name, header=None)
    else:
        df = pd.read_csv(source, header=None)

    n_before = len(df)
    df_clean = df.dropna(how="any").reset_index(drop=True)
    n_after = len(df_clean)
    n_discarded = n_before - n_after
    if n_discarded > 0:
        print(f"  [aviso] Se descartaron {n_discarded} bloque(s) incompleto(s) "
              f"de un total de {n_before}.")

    return df_clean.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 2. Calculo de metricas por bloque
# ---------------------------------------------------------------------------

def per_block_success_rate(matrix):
    """Tasa de exito por bloque: promedio de aciertos (0/1) en el bloque."""
    return matrix.mean(axis=1)


def per_block_switch_probability(matrix):
    """Probabilidad de cambio por bloque: promedio de cambios (0/1) en el bloque."""
    return matrix.mean(axis=1)


def per_block_learning_rate(matrix):
    """
    Tasa de aprendizaje por bloque, operacionalizada como la pendiente de una
    regresion lineal simple de la tasa de exito (0/1) en funcion de la
    posicion del ensayo dentro del bloque (0, 1, ..., n_trials-1).

    NOTA METODOLOGICA: esta es una de varias operacionalizaciones posibles de
    "derivada de la tasa de exito". Si se prefiere una version mas simple
    (diferencia de dos puntos), reemplazar el cuerpo de esta funcion por:
        return (matrix[:, -1] - matrix[:, 0]) / (matrix.shape[1] - 1)
    Ambas dan resultados cualitativamente similares; se eligio la pendiente
    de regresion por ser mas robusta a ruido en ensayos individuales.
    """
    n_trials = matrix.shape[1]
    t = np.arange(n_trials)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    # Pendiente OLS vectorizada para todas las filas a la vez:
    # slope = sum((t - t_mean) * (y - y_mean)) / sum((t - t_mean)^2)
    y = matrix
    y_mean = y.mean(axis=1, keepdims=True)
    slopes = (t_centered * (y - y_mean)).sum(axis=1) / denom
    return slopes


METRIC_FUNCS = {
    "success": per_block_success_rate,
    "switch": per_block_switch_probability,
    "learning_rate": per_block_learning_rate,
}


# ---------------------------------------------------------------------------
# 3. Bootstrap TOST power analysis
# ---------------------------------------------------------------------------

def bootstrap_tost_power(x1, x2_pool, n1, n2, margin, alpha, n_sims=3000,
                          assume_null_true=False, rng=None):
    """
    Estima, mediante bootstrap no parametrico, la potencia de una prueba TOST
    para un tamano muestral de simulacion n2 dado.

    Parameters
    ----------
    x1 : array-like
        Datos reales completos de los NHP para esta metrica/fase (el "pool"
        del cual se remuestrea el lado de referencia).
    x2_pool : array-like
        Datos reales (piloto) del modelo para esta metrica/fase (el "pool"
        del cual se remuestrea el lado del modelo).
    n1 : int
        Tamano de la muestra de NHP a usar en cada remuestreo (tipicamente,
        el numero real de bloques NHP disponibles para esta fase).
    n2 : int
        Tamano de la muestra simulada candidata a evaluar.
    margin : float
        Margen de equivalencia (SESOI en unidades crudas); los limites TOST
        seran [-margin, +margin].
    alpha : float
        Nivel de significancia (ya corregido por comparaciones multiples).
    n_sims : int
        Numero de replicas de bootstrap (mientras mas, mas precisa la
        estimacion de potencia, a costo de tiempo de computo).
    assume_null_true : bool
        Si True, el pool del modelo se recentra para que su media coincida
        exactamente con la del NHP antes de remuestrear, aislando la
        potencia debida solo al ruido de muestreo (escenario "el modelo es
        perfectamente equivalente"). Si False (por defecto), se remuestrea
        directamente desde los datos piloto del modelo tal como fueron
        observados, lo que refleja la potencia esperada dado el desempeno
        actual (posiblemente imperfecto) del modelo piloto.
    rng : numpy.random.Generator, optional

    Returns
    -------
    float
        Potencia estimada (proporcion de replicas en que TOST concluyo
        equivalencia).
    """
    if rng is None:
        rng = np.random.default_rng()

    x1 = np.asarray(x1)
    x2_pool = np.asarray(x2_pool)

    if assume_null_true:
        shift = x1.mean() - x2_pool.mean()
        x2_pool = x2_pool + shift

    rejections = 0
    for _ in range(n_sims):
        s1 = rng.choice(x1, size=n1, replace=True)
        s2 = rng.choice(x2_pool, size=n2, replace=True)
        p_overall, _, _ = ttost_ind(s2, s1, -margin, margin, usevar="unequal")
        if p_overall < alpha:
            rejections += 1

    return rejections / n_sims


def find_min_n2_bootstrap(x1, x2_pool, n1, margin, alpha, target_power=0.80,
                           n_sims=3000, assume_null_true=False,
                           lo=30, hi=20000, tol=15, seed=None):
    """
    Busqueda por biseccion del n2 minimo (numero de bloques simulados) que
    alcanza la potencia objetivo, usando bootstrap_tost_power como funcion
    de evaluacion.

    Returns
    -------
    int
        n2 minimo estimado. Si ni siquiera `hi` alcanza la potencia objetivo,
        se retorna `hi` junto con una advertencia impresa en pantalla (esto
        senala una posible limitacion estructural de potencia, como ocurre
        en la fase DBS dado su reducido n1).
    """
    rng = np.random.default_rng(seed)

    power_hi = bootstrap_tost_power(x1, x2_pool, n1, hi, margin, alpha,
                                     n_sims=n_sims,
                                     assume_null_true=assume_null_true, rng=rng)
    if power_hi < target_power:
        print(f"  [ADVERTENCIA] Incluso con n2={hi}, la potencia estimada "
              f"({power_hi:.3f}) no alcanza el objetivo ({target_power}). "
              f"Esto puede indicar un limite estructural de potencia dado "
              f"el n1 disponible (ver fase DBS). Considere ampliar el "
              f"margen de equivalencia.")
        return hi, power_hi

    while hi - lo > tol:
        mid = (lo + hi) // 2
        power_mid = bootstrap_tost_power(x1, x2_pool, n1, mid, margin, alpha,
                                          n_sims=n_sims,
                                          assume_null_true=assume_null_true,
                                          rng=rng)
        if power_mid >= target_power:
            hi = mid
        else:
            lo = mid

    final_power = bootstrap_tost_power(x1, x2_pool, n1, hi, margin, alpha,
                                        n_sims=n_sims,
                                        assume_null_true=assume_null_true,
                                        rng=rng)
    return hi, final_power
