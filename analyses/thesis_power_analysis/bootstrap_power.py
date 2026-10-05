"""
Analisis de potencia estadistica a priori para las pruebas de equivalencia (TOST)
entre el modelo computacional y los datos de los NHP, mediante bootstrap no
parametrico.

Estructura esperada de los datos crudos (por fase y por tipo de dato):
    - Cada fila es un bloque.
    - Cada columna es un ensayo (trial) dentro del bloque (hasta 15).
    - Un bloque puede terminar antes de llegar a 15 ensayos (tanto en los
      datos NHP como en el modelo, bajo la regla "nhp_like": el bloque
      termina apenas se alcanza el umbral de exito, que puede ocurrir
      antes de 15 ensayos). Esas filas tienen celdas vacias (NaN) al
      FINAL y SE CONSERVAN: las metricas por bloque se calculan solo
      sobre los ensayos reales disponibles en esa fila (ver
      per_block_success_rate, per_block_switch_probability,
      per_block_learning_rate). Solo se descarta una fila si esta
      COMPLETAMENTE vacia (todas sus celdas NaN), lo cual no deberia
      ocurrir dado el largo minimo de bloque de la tarea, pero se revisa
      igual de forma defensiva.
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
    en memoria. Un bloque mas corto que el ancho de la hoja (NaN al final de
    su fila, porque termino antes de llegar a esa cantidad de ensayos) SE
    CONSERVA -- las funciones per_block_* de mas abajo saben calcular sus
    metricas usando solo los ensayos reales de cada fila. Solo se descarta
    una fila si esta COMPLETAMENTE vacia (todas sus celdas NaN).

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
        Matriz de forma (n_bloques_utilizables, n_ensayos), dtype float.
        Puede contener NaN al final de una fila para bloques mas cortos
        que n_ensayos; los consumidores (per_block_*) las manejan.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif str(source).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(source, sheet_name=sheet_name, header=None)
    else:
        df = pd.read_csv(source, header=None)

    n_before = len(df)
    df_clean = df.dropna(how="all").reset_index(drop=True)
    n_after = len(df_clean)
    n_discarded = n_before - n_after
    if n_discarded > 0:
        print(f"  [aviso] Se descartaron {n_discarded} bloque(s) completamente "
              f"vacio(s) de un total de {n_before}.")

    return df_clean.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 2. Calculo de metricas por bloque
# ---------------------------------------------------------------------------

def per_block_success_rate(matrix):
    """Tasa de exito por bloque: promedio de aciertos (0/1) en el bloque.

    Usa nanmean para que un bloque mas corto que el ancho de la matriz
    (NaN al final de su fila, ver load_trial_matrix) se promedie solo
    sobre sus ensayos reales, en vez de ser descartado."""
    return np.nanmean(matrix, axis=1)


def per_block_switch_probability(matrix):
    """Probabilidad de cambio por bloque: promedio de cambios (0/1) en el
    bloque. Ver nota de nanmean en per_block_success_rate."""
    return np.nanmean(matrix, axis=1)


def per_block_learning_rate(matrix):
    """
    Tasa de aprendizaje por bloque, operacionalizada como la diferencia de
    dos puntos: (ultimo ensayo real - primer ensayo real) dividido por
    (n_validos - 1), de la tasa de exito (0/1) a lo largo del bloque.

    NOTA METODOLOGICA: esta es una de varias operacionalizaciones posibles
    de "derivada de la tasa de exito". Una alternativa es la pendiente de
    una regresion lineal simple (mas robusta a ruido en ensayos
    individuales, a costa de ser menos directa de interpretar); si se
    prefiere esa version, reemplazar el cuerpo de esta funcion por una
    regresion OLS fila por fila sobre los ensayos validos de cada bloque.

    Un bloque mas corto que el ancho de la matriz (NaN al final de su fila)
    se evalua usando solo sus ensayos reales -- no se asume que todas las
    filas tengan la misma cantidad de ensayos validos, por lo que esto ya
    NO puede vectorizarse directamente y se calcula fila por fila. Si una
    fila tiene menos de 2 ensayos reales (no deberia ocurrir dado el largo
    minimo de bloque de la tarea, ~12 ensayos), se devuelve 0.0 (sin
    tendencia) en lugar de NaN, para no propagar NaN silenciosamente a los
    promedios/agregados posteriores.
    """
    n_rows = matrix.shape[0]
    rates = np.empty(n_rows)
    for i in range(n_rows):
        row = matrix[i]
        valid = ~np.isnan(row)
        n_valid = int(valid.sum())
        if n_valid < 2:
            rates[i] = 0.0
            continue
        y = row[valid]
        rates[i] = (y[-1] - y[0]) / (n_valid - 1)
    return rates


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
                           lo=30, hi=50000, tol=15, seed=None):
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
