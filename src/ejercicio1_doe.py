"""Ejercicio 1: ajuste de la temperatura inicial del Recocido Simulado
mediante un diseño factorial completo 3 x 3.

Factores
--------
- Temperatura inicial T0: 3 niveles (bajo, medio, alto) calculados como
  fracción del rango [T_MIN, T_MAX] usando un único r ~ U(0, 1) sembrado
  con la matrícula.
- Tamaño del TSP: n in {5, 10, 15} (una instancia fija por tamaño).
- 10 réplicas por celda (semillas distintas) -> 90 corridas.

Salidas
-------
- datos/resultados.csv          : una fila por corrida.
- datos/diseno.json             : r, fracciones, temperaturas y parámetros.
- datos/instancias.json         : coordenadas de las tres instancias.
- datos/anova.csv               : ANOVA de dos factores (tipo II).
- datos/anova_por_tamano.csv    : ANOVA de un factor por tamaño.
- datos/tukey_por_tamano.csv    : comparaciones múltiples de Tukey.
- datos/bootstrap_puntos.csv    : mediana e IC 95 % por punto de diseño.
- datos/mejor_por_tamano.csv    : mejor temperatura por tamaño.
- graficas/*.png                : figuras del reporte.
"""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

from sa_tsp import (
    generate_instance,
    random_initial_route,
    simulated_annealing,
    tour_length,
)

# ---------------------------------------------------------------------------
# Parámetros del experimento
# ---------------------------------------------------------------------------
MATRICULA = 2177628
T_MIN, T_MAX = 1.0, 100.0
TAMANOS = (5, 10, 15)
NIVELES = ("bajo", "medio", "alto")
REPLICAS = 10
MAX_ITER = 500
COOLING = 0.9985
ALPHA = 0.05
BOOTSTRAP_REPS = 10_000
PRESUPUESTOS = (100, 200, 500, 1_000, 5_000)
CALIB_REPS = 5

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_GRAFICAS = RAIZ / "graficas"

COLORES_TAMANO = {5: "#2a9d8f", 10: "#e76f51", 15: "#264653"}
COLOR_PRINCIPAL = "#1f4e79"


# ---------------------------------------------------------------------------
# Diseño experimental
# ---------------------------------------------------------------------------
def calcular_r(matricula: int) -> float:
    """Genera el único valor r ~ U(0, 1) usado en todo el experimento."""
    return random.Random(matricula).random()


def construir_niveles(r: float) -> dict[str, dict[str, float]]:
    """Convierte las fracciones de la Opción A a temperaturas reales."""
    fracciones = {
        "bajo": 0.3 * r,
        "medio": 0.3 * r + 0.3,
        "alto": 0.3 * r + 0.6,
    }
    return {
        nombre: {
            "fraccion": fraccion,
            "temperatura": T_MIN + fraccion * (T_MAX - T_MIN),
        }
        for nombre, fraccion in fracciones.items()
    }


def semilla_corrida(n: int, indice_nivel: int, replica: int) -> int:
    """Semilla determinista y distinta para cada corrida."""
    return MATRICULA + 1_000 * n + 100 * indice_nivel + replica


# ---------------------------------------------------------------------------
# Ejecución del experimento
# ---------------------------------------------------------------------------
def ejecutar_experimento(niveles: dict[str, dict[str, float]]):
    filas = []
    instancias = {}

    for n in TAMANOS:
        points = generate_instance(n, MATRICULA + n)
        instancias[str(n)] = [[x, y] for x, y in points]
        print(f"\nInstancia n={n}: {n} ciudades (semilla {MATRICULA + n})")

        for indice_nivel, nombre in enumerate(NIVELES):
            T0 = niveles[nombre]["temperatura"]
            costos = []
            for replica in range(REPLICAS):
                semilla = semilla_corrida(n, indice_nivel, replica)
                rng = random.Random(semilla)
                ruta_inicial = random_initial_route(n, rng)
                resultado = simulated_annealing(
                    points,
                    ruta_inicial,
                    T0,
                    max_iter=MAX_ITER,
                    cooling=COOLING,
                    rng=rng,
                )
                costos.append(resultado.best_cost)
                filas.append(
                    {
                        "tamano": n,
                        "nivel": nombre,
                        "temperatura": T0,
                        "replica": replica + 1,
                        "semilla": semilla,
                        "costo": resultado.best_cost,
                        "iteraciones": resultado.iterations,
                        "tiempo_ms": resultado.elapsed_ms,
                        "ruta_inicial": " ".join(map(str, ruta_inicial)),
                        "mejor_ruta": " ".join(map(str, resultado.best_route)),
                    }
                )
            print(
                f"  T0={T0:6.2f} ({nombre:<5}) -> "
                f"media={np.mean(costos):7.2f}  sd={np.std(costos, ddof=1):6.2f}"
            )

    df = pd.DataFrame(filas)
    df["nivel"] = pd.Categorical(df["nivel"], categories=NIVELES, ordered=True)
    return df, instancias


# ---------------------------------------------------------------------------
# ANOVA
# ---------------------------------------------------------------------------
def anova_dos_factores(df: pd.DataFrame):
    modelo = ols("costo ~ C(nivel) * C(tamano)", data=df).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)
    tabla.index = ["temperatura", "tamano", "interaccion", "error"]
    tabla = tabla.rename(columns={"PR(>F)": "p_value"})

    ss_error = float(tabla.loc["error", "sum_sq"])
    tabla["eta2_parcial"] = np.nan
    for fuente in ("temperatura", "tamano", "interaccion"):
        ss = float(tabla.loc[fuente, "sum_sq"])
        tabla.loc[fuente, "eta2_parcial"] = ss / (ss + ss_error)
    tabla["significativo"] = tabla["p_value"] < ALPHA

    residuos = modelo.resid
    grupos = [
        grupo["costo"].to_numpy()
        for _, grupo in df.groupby(["tamano", "nivel"], observed=True)
    ]
    shapiro_stat, shapiro_p = stats.shapiro(residuos)
    levene_stat, levene_p = stats.levene(*grupos)
    brown_stat, brown_p = stats.levene(*grupos, center="median")

    diagnostico = {
        "shapiro_W": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "levene_F": float(levene_stat),
        "levene_p": float(levene_p),
        "brown_forsythe_F": float(brown_stat),
        "brown_forsythe_p": float(brown_p),
        "r2": float(modelo.rsquared),
        "r2_ajustado": float(modelo.rsquared_adj),
    }
    return modelo, tabla, diagnostico


def anova_por_tamano(df: pd.DataFrame):
    """ANOVA de un factor, Tukey, Kruskal-Wallis y Welch por cada tamaño.

    En n = 5 las 30 corridas alcanzan exactamente el mismo óptimo, por lo
    que la varianza dentro de cada celda es cero y las pruebas clásicas no
    están definidas. En ese caso se reporta explícitamente la ausencia de
    variación en lugar de un F o un p-value espurio.
    """
    filas_anova = []
    filas_tukey = []
    filas_kruskal = []
    filas_welch = []

    for n in TAMANOS:
        sub = df[df["tamano"] == n]
        grupos = [
            sub[sub["nivel"] == nombre]["costo"].to_numpy() for nombre in NIVELES
        ]
        sin_variacion = float(np.std(sub["costo"], ddof=1)) < 1e-9

        if sin_variacion:
            nota = "Sin variación: las 10 réplicas alcanzaron el mismo costo."
            for fuente in ("temperatura", "error"):
                filas_anova.append(
                    {
                        "tamano": n,
                        "fuente": fuente,
                        "sum_sq": 0.0,
                        "gl": 2.0 if fuente == "temperatura" else 27.0,
                        "F": np.nan,
                        "p_value": np.nan,
                        "nota": nota,
                    }
                )
            tukey_n5 = []
            for grupo1, grupo2 in combinations(NIVELES, 2):
                tukey_n5.append(
                    {
                        "tamano": n,
                        "group1": grupo1,
                        "group2": grupo2,
                        "meandiff": 0.0,
                        "p-adj": np.nan,
                        "lower": np.nan,
                        "upper": np.nan,
                        "reject": False,
                        "nota": nota,
                    }
                )
            filas_tukey.append(pd.DataFrame(tukey_n5))
            filas_kruskal.append(
                {"tamano": n, "H": np.nan, "p_value": np.nan, "nota": nota}
            )
            filas_welch.append(
                {"tamano": n, "F": np.nan, "p_value": np.nan, "nota": nota}
            )
            continue

        modelo = ols("costo ~ C(nivel)", data=sub).fit()
        tabla = sm.stats.anova_lm(modelo, typ=2)
        tabla.index = ["temperatura", "error"]
        for fuente, fila in tabla.iterrows():
            filas_anova.append(
                {
                    "tamano": n,
                    "fuente": fuente,
                    "sum_sq": float(fila["sum_sq"]),
                    "gl": float(fila["df"]),
                    "F": float(fila["F"]) if fuente != "error" else np.nan,
                    "p_value": (
                        float(fila["PR(>F)"]) if fuente != "error" else np.nan
                    ),
                    "nota": "",
                }
            )

        tukey = pairwise_tukeyhsd(
            sub["costo"].to_numpy(),
            sub["nivel"].astype(str).to_numpy(),
            alpha=ALPHA,
        )
        resumen = tukey.summary()
        tabla_tukey = pd.DataFrame(resumen.data[1:], columns=resumen.data[0])
        tabla_tukey.insert(0, "tamano", n)
        tabla_tukey["nota"] = ""
        filas_tukey.append(tabla_tukey)

        h_stat, h_p = stats.kruskal(*grupos)
        filas_kruskal.append(
            {
                "tamano": n,
                "H": float(h_stat),
                "p_value": float(h_p),
                "nota": "",
            }
        )

        welch = anova_oneway(grupos, use_var="unequal")
        filas_welch.append(
            {
                "tamano": n,
                "F": float(welch.statistic),
                "p_value": float(welch.pvalue),
                "nota": "",
            }
        )

    return (
        pd.DataFrame(filas_anova),
        pd.concat(filas_tukey, ignore_index=True),
        pd.DataFrame(filas_kruskal),
        pd.DataFrame(filas_welch),
    )


def diagnostico_robustez(df: pd.DataFrame):
    """Repite el ANOVA excluyendo n = 5 y revisa los supuestos.

    n = 5 produce varianza cero en todas las celdas (efecto techo), lo que
    infla las pruebas de normalidad y homocedasticidad del modelo completo.
    """
    sub = df[df["tamano"] > 5].copy()
    modelo = ols("costo ~ C(nivel) * C(tamano)", data=sub).fit()
    tabla = sm.stats.anova_lm(modelo, typ=2)
    tabla.index = ["temperatura", "tamano", "interaccion", "error"]
    tabla = tabla.rename(columns={"PR(>F)": "p_value"})

    grupos = [
        grupo["costo"].to_numpy()
        for _, grupo in sub.groupby(["tamano", "nivel"], observed=True)
    ]
    shapiro_stat, shapiro_p = stats.shapiro(modelo.resid)
    levene_stat, levene_p = stats.levene(*grupos)
    diagnostico = {
        "shapiro_W_sin_n5": float(shapiro_stat),
        "shapiro_p_sin_n5": float(shapiro_p),
        "levene_F_sin_n5": float(levene_stat),
        "levene_p_sin_n5": float(levene_p),
    }
    return tabla, diagnostico


def calibrar_presupuesto(niveles: dict[str, dict[str, float]]):
    """Corrida preliminar para justificar el presupuesto de 500 iteraciones.

    Usa 5 réplicas por celda (no forma parte del diseño de 90 corridas) y
    barre el número de iteraciones. Guarda el costo para mostrar que con
    5 000 iteraciones todas las temperaturas convergen al mismo óptimo.
    """
    filas = []
    for max_iter in PRESUPUESTOS:
        for n in TAMANOS:
            points = generate_instance(n, MATRICULA + n)
            for indice_nivel, nombre in enumerate(NIVELES):
                T0 = niveles[nombre]["temperatura"]
                for replica in range(CALIB_REPS):
                    semilla = semilla_corrida(n, indice_nivel, replica)
                    rng = random.Random(semilla)
                    ruta_inicial = random_initial_route(n, rng)
                    resultado = simulated_annealing(
                        points,
                        ruta_inicial,
                        T0,
                        max_iter=max_iter,
                        cooling=COOLING,
                        rng=rng,
                    )
                    filas.append(
                        {
                            "max_iter": max_iter,
                            "tamano": n,
                            "nivel": nombre,
                            "temperatura": T0,
                            "replica": replica + 1,
                            "costo": resultado.best_cost,
                        }
                    )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Bootstrap de la mediana
# ---------------------------------------------------------------------------
def bootstrap_mediana(
    valores: np.ndarray, n_boot: int = BOOTSTRAP_REPS, semilla: int = 0
) -> tuple[float, float, float]:
    rng = np.random.default_rng(semilla)
    indices = rng.integers(0, len(valores), size=(n_boot, len(valores)))
    medianas = np.median(valores[indices], axis=1)
    return (
        float(np.median(valores)),
        float(np.percentile(medianas, 2.5)),
        float(np.percentile(medianas, 97.5)),
    )


def analisis_bootstrap(df: pd.DataFrame):
    filas = []
    for n in TAMANOS:
        for indice_nivel, nombre in enumerate(NIVELES):
            valores = df[
                (df["tamano"] == n) & (df["nivel"] == nombre)
            ]["costo"].to_numpy()
            mediana, li, ls = bootstrap_mediana(
                valores,
                semilla=MATRICULA + 7_919 * (n * 10 + indice_nivel),
            )
            filas.append(
                {
                    "tamano": n,
                    "nivel": nombre,
                    "temperatura": float(df[df["nivel"] == nombre]["temperatura"].iloc[0]),
                    "n": len(valores),
                    "mediana": mediana,
                    "ic95_inf": li,
                    "ic95_sup": ls,
                    "media": float(np.mean(valores)),
                    "sd": float(np.std(valores, ddof=1)),
                    "min": float(np.min(valores)),
                    "max": float(np.max(valores)),
                }
            )
    return pd.DataFrame(filas)


def seleccionar_mejor(bootstrap_df: pd.DataFrame):
    filas = []
    for n in TAMANOS:
        sub = bootstrap_df[bootstrap_df["tamano"] == n].copy()
        mejor_mediana = sub["mediana"].min()
        candidatos = sub[np.isclose(sub["mediana"], mejor_mediana, atol=1e-6)]
        candidatos = candidatos.sort_values("temperatura")
        elegido = candidatos.iloc[0]
        filas.append(
            {
                "tamano": n,
                "nivel_optimo": elegido["nivel"],
                "temperatura_optima": elegido["temperatura"],
                "mediana_bootstrap": elegido["mediana"],
                "ic95_inf": elegido["ic95_inf"],
                "ic95_sup": elegido["ic95_sup"],
                "empate": len(candidatos) > 1,
                "niveles_empatados": ", ".join(candidatos["nivel"].astype(str)),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _intervalo_pooled(mse: float, gl_error: int, n_obs: int) -> float:
    """Semiintervalo de confianza usando el error pooled del ANOVA."""
    return float(stats.t.ppf(0.975, gl_error) * np.sqrt(mse / n_obs))


def graficar_efectos_principales(
    df: pd.DataFrame, niveles, mse: float, gl_error: int, ruta: Path
):
    etiquetas = [
        f"{nombre}\nT={niveles[nombre]['temperatura']:.2f}"
        for nombre in NIVELES
    ]
    medias_temp = df.groupby("nivel", observed=True)["costo"].mean().reindex(NIVELES)
    errores_temp = [
        _intervalo_pooled(mse, gl_error, int((df["nivel"] == nombre).sum()))
        for nombre in NIVELES
    ]
    medias_n = df.groupby("tamano")["costo"].mean()
    errores_n = [
        _intervalo_pooled(mse, gl_error, int((df["tamano"] == n).sum()))
        for n in TAMANOS
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(NIVELES))
    axes[0].errorbar(
        x,
        medias_temp,
        yerr=errores_temp,
        fmt="o-",
        color=COLOR_PRINCIPAL,
        capsize=6,
        lw=2.2,
        ms=8,
    )
    axes[0].set_xticks(x, etiquetas)
    axes[0].set_xlabel("Nivel de temperatura inicial")
    axes[0].set_ylabel("Costo medio del tour (menor es mejor)")
    axes[0].set_title("Efecto principal: temperatura")
    axes[0].grid(True, ls=":", alpha=0.65)

    axes[1].errorbar(
        TAMANOS,
        medias_n,
        yerr=errores_n,
        fmt="s-",
        color="#b5651d",
        capsize=6,
        lw=2.2,
        ms=8,
    )
    axes[1].set_xticks(TAMANOS)
    axes[1].set_xlabel("Tamaño del TSP (n)")
    axes[1].set_ylabel("Costo medio del tour (menor es mejor)")
    axes[1].set_title("Efecto principal: tamaño")
    axes[1].grid(True, ls=":", alpha=0.65)

    fig.suptitle(
        "Efectos principales del diseño factorial 3×3 (medias ± IC 95 %)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_interaccion(
    df: pd.DataFrame, niveles, mse: float, gl_error: int, ruta: Path
):
    etiquetas = [
        f"{nombre}\nT={niveles[nombre]['temperatura']:.2f}" for nombre in NIVELES
    ]
    x = np.arange(len(NIVELES))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for n in TAMANOS:
        sub = df[df["tamano"] == n]
        medias = sub.groupby("nivel", observed=True)["costo"].mean().reindex(NIVELES)
        errores = [
            _intervalo_pooled(
                mse, gl_error, int((sub["nivel"] == nombre).sum())
            )
            for nombre in NIVELES
        ]
        ax.errorbar(
            x,
            medias,
            yerr=errores,
            fmt="o-",
            label=f"n = {n}",
            color=COLORES_TAMANO[n],
            capsize=5,
            lw=2,
            ms=7,
        )

    ax.set_xticks(x, etiquetas)
    ax.set_xlabel("Nivel de temperatura inicial")
    ax.set_ylabel("Costo medio del tour (menor es mejor)")
    ax.set_title(
        "Interacción temperatura × tamaño (medias ± IC 95 %)", fontweight="bold"
    )
    ax.legend(title="Tamaño", frameon=True)
    ax.grid(True, ls=":", alpha=0.65)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_boxplots(df: pd.DataFrame, niveles, ruta: Path):
    etiquetas = [
        f"{nombre}\nT={niveles[nombre]['temperatura']:.2f}" for nombre in NIVELES
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    rng = np.random.default_rng(MATRICULA)

    for ax, n in zip(axes, TAMANOS):
        sub = df[df["tamano"] == n]
        datos = [sub[sub["nivel"] == nombre]["costo"].to_numpy() for nombre in NIVELES]
        cajas = ax.boxplot(
            datos,
            tick_labels=etiquetas,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "lw": 1.8},
        )
        for caja in cajas["boxes"]:
            caja.set_facecolor(COLORES_TAMANO[n])
            caja.set_alpha(0.35)
        for i, valores in enumerate(datos, start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(valores))
            ax.scatter(
                i + jitter,
                valores,
                s=28,
                color=COLORES_TAMANO[n],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Nivel de temperatura")
        ax.set_ylabel("Costo final del tour")
        ax.grid(True, axis="y", ls=":", alpha=0.6)

    fig.suptitle(
        "Distribución del costo por temperatura y tamaño (10 réplicas)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_bootstrap(bootstrap_df: pd.DataFrame, niveles, ruta: Path):
    etiquetas = [
        f"{nombre}\nT={niveles[nombre]['temperatura']:.2f}" for nombre in NIVELES
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    x = np.arange(len(NIVELES))

    for ax, n in zip(axes, TAMANOS):
        sub = (
            bootstrap_df[bootstrap_df["tamano"] == n]
            .set_index("nivel")
            .reindex(NIVELES)
        )
        medianas = sub["mediana"].to_numpy()
        inferiores = medianas - sub["ic95_inf"].to_numpy()
        superiores = sub["ic95_sup"].to_numpy() - medianas
        ax.errorbar(
            x,
            medianas,
            yerr=np.vstack([inferiores, superiores]),
            fmt="o",
            color=COLORES_TAMANO[n],
            capsize=6,
            ms=8,
            lw=2,
        )
        sin_variacion = bool(
            (sub["ic95_sup"] - sub["ic95_inf"]).abs().max() < 1e-6
        )
        if sin_variacion:
            ax.text(
                0.5,
                0.92,
                f"{medianas[0]:.1f} en los tres niveles\n(sin variación)",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#999999"},
            )
        else:
            for indice, (xi, mediana, li, ls) in enumerate(
                zip(x, medianas, sub["ic95_inf"], sub["ic95_sup"])
            ):
                if indice < 2:
                    desplazamiento, alineacion = (10, 0), "left"
                else:
                    desplazamiento, alineacion = (8, 9), "left"
                ax.annotate(
                    f"{mediana:.1f}\n[{li:.1f}, {ls:.1f}]",
                    (xi, mediana),
                    textcoords="offset points",
                    xytext=desplazamiento,
                    ha=alineacion,
                    va="center",
                    fontsize=7.5,
                )
        ax.set_xticks(x, etiquetas)
        ax.set_xlim(-0.35, 3.05)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("Nivel de temperatura")
        ax.set_ylabel("Mediana del costo (IC 95 % bootstrap)")
        ax.grid(True, ls=":", alpha=0.6)
        rango = float(np.max(sub["ic95_sup"]) - np.min(sub["ic95_inf"]))
        margen = max(0.26 * rango, 0.02 * float(np.median(medianas)), 0.5)
        ax.set_ylim(medianas.min() - margen, medianas.max() + margen)

    fig.suptitle(
        "Medianas bootstrap e intervalos de confianza del 95 % por punto de diseño",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_residuos(modelo, ruta: Path):
    """Q-Q normal y residuos vs. ajustados para revisar supuestos del ANOVA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    stats.probplot(modelo.resid, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q normal de los residuos")
    axes[0].set_xlabel("Cuantiles teóricos")
    axes[0].set_ylabel("Cuantiles observados")
    axes[0].grid(True, ls=":", alpha=0.6)

    axes[1].scatter(
        modelo.fittedvalues,
        modelo.resid,
        s=32,
        alpha=0.75,
        color=COLOR_PRINCIPAL,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].axhline(0, color="black", lw=1.2)
    axes[1].set_title("Residuos vs. valores ajustados")
    axes[1].set_xlabel("Valor ajustado")
    axes[1].set_ylabel("Residuo")
    axes[1].grid(True, ls=":", alpha=0.6)

    fig.suptitle(
        "Diagnóstico de los supuestos del ANOVA de dos factores",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_instancias(df: pd.DataFrame, instancias, ruta: Path):
    """Dibuja las tres instancias y el mejor tour encontrado en el experimento."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    for ax, n in zip(axes, TAMANOS):
        puntos = np.asarray(instancias[str(n)], dtype=float)
        mejor_fila = df[df["tamano"] == n].sort_values("costo").iloc[0]
        ruta_tour = [int(v) for v in mejor_fila["mejor_ruta"].split()]
        ruta_tour.append(ruta_tour[0])
        ax.plot(
            puntos[ruta_tour, 0],
            puntos[ruta_tour, 1],
            "-",
            color=COLORES_TAMANO[n],
            lw=1.6,
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            puntos[:, 0],
            puntos[:, 1],
            s=60,
            color=COLORES_TAMANO[n],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for indice, (x, y) in enumerate(puntos):
            ax.annotate(
                str(indice),
                (x, y),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="#333333",
            )
        ax.set_title(
            f"n = {n} · mejor costo = {mejor_fila['costo']:.2f}", fontsize=10
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, ls=":", alpha=0.55)

    fig.suptitle(
        "Instancias TSP y mejor tour encontrado (una instancia por tamaño)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


def graficar_calibracion(calibracion: pd.DataFrame, ruta: Path):
    """Muestra cómo el efecto de la temperatura desaparece al aumentar las iteraciones."""
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for n in TAMANOS:
        sub = calibracion[calibracion["tamano"] == n]
        agregado = sub.groupby("max_iter")["costo"].agg(
            media="mean", minimo="min", maximo="max"
        )
        ax.plot(
            agregado.index,
            agregado["media"],
            "o-",
            color=COLORES_TAMANO[n],
            lw=2,
            ms=6,
            label=f"n = {n}",
        )
        ax.fill_between(
            agregado.index,
            agregado["minimo"],
            agregado["maximo"],
            color=COLORES_TAMANO[n],
            alpha=0.13,
        )

    ax.axvline(MAX_ITER, color="#777777", ls="--", lw=1.2)
    ax.annotate(
        "presupuesto elegido\n(500 iteraciones)",
        xy=(MAX_ITER, ax.get_ylim()[1]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )
    ax.set_xscale("log")
    ax.set_xticks(PRESUPUESTOS, labels=[str(v) for v in PRESUPUESTOS])
    ax.set_xlabel("Número de iteraciones (escala logarítmica)")
    ax.set_ylabel("Costo final (media; banda: mín–máx entre temperaturas)")
    ax.set_title(
        "Calibración del presupuesto: convergencia y sensibilidad a la temperatura",
        fontweight="bold",
    )
    ax.legend(title="Tamaño", frameon=True)
    ax.grid(True, ls=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    DIR_DATOS.mkdir(exist_ok=True)
    DIR_GRAFICAS.mkdir(exist_ok=True)

    r = calcular_r(MATRICULA)
    niveles = construir_niveles(r)

    print("=" * 72)
    print("DISEÑO FACTORIAL 3x3 - TEMPERATURA DEL RECOCIDO SIMULADO")
    print("=" * 72)
    print(f"Matrícula (semilla): {MATRICULA}")
    print(f"r ~ U(0,1)         : {r!r}")
    for nombre in NIVELES:
        print(
            f"  {nombre:<5}: fracción={niveles[nombre]['fraccion']:.6f}  "
            f"T={niveles[nombre]['temperatura']:.4f}"
        )
    print(f"Rango de temperatura: [{T_MIN}, {T_MAX}]")
    print(f"Presupuesto         : {MAX_ITER} iteraciones, cooling={COOLING}")
    print(f"Réplicas por celda  : {REPLICAS}")

    df, instancias = ejecutar_experimento(niveles)

    df.to_csv(DIR_DATOS / "resultados.csv", index=False, encoding="utf-8")
    with open(DIR_DATOS / "diseno.json", "w", encoding="utf-8") as archivo:
        json.dump(
            {
                "matricula": MATRICULA,
                "r": r,
                "t_min": T_MIN,
                "t_max": T_MAX,
                "niveles": niveles,
                "tamanos": list(TAMANOS),
                "replicas": REPLICAS,
                "max_iter": MAX_ITER,
                "cooling": COOLING,
                "alpha": ALPHA,
                "bootstrap_reps": BOOTSTRAP_REPS,
            },
            archivo,
            indent=2,
            ensure_ascii=False,
        )
    with open(DIR_DATOS / "instancias.json", "w", encoding="utf-8") as archivo:
        json.dump(instancias, archivo, indent=2)

    modelo, tabla_anova, diagnostico = anova_dos_factores(df)
    tabla_anova.to_csv(DIR_DATOS / "anova.csv", encoding="utf-8")
    anova_tamano, tukey, kruskal, welch = anova_por_tamano(df)
    anova_tamano.to_csv(DIR_DATOS / "anova_por_tamano.csv", index=False)
    tukey.to_csv(DIR_DATOS / "tukey_por_tamano.csv", index=False)
    kruskal.to_csv(DIR_DATOS / "kruskal_por_tamano.csv", index=False)
    welch.to_csv(DIR_DATOS / "welch_por_tamano.csv", index=False)

    tabla_sin_n5, diagnostico_sin_n5 = diagnostico_robustez(df)
    tabla_sin_n5.to_csv(DIR_DATOS / "anova_sin_n5.csv", encoding="utf-8")
    diagnostico.update(diagnostico_sin_n5)

    bootstrap_df = analisis_bootstrap(df)
    bootstrap_df.to_csv(DIR_DATOS / "bootstrap_puntos.csv", index=False)
    mejor = seleccionar_mejor(bootstrap_df)
    mejor.to_csv(DIR_DATOS / "mejor_por_tamano.csv", index=False)

    with open(DIR_DATOS / "diagnostico.json", "w", encoding="utf-8") as archivo:
        json.dump(diagnostico, archivo, indent=2)

    calibracion = calibrar_presupuesto(niveles)
    calibracion.to_csv(DIR_DATOS / "calibracion.csv", index=False)
    resumen_calibracion = (
        calibracion.groupby(["max_iter", "tamano"], observed=True)["costo"]
        .agg(media="mean", minimo="min", maximo="max")
        .reset_index()
    )
    resumen_calibracion.to_csv(
        DIR_DATOS / "calibracion_resumen.csv", index=False
    )

    mse = float(
        tabla_anova.loc["error", "sum_sq"] / tabla_anova.loc["error", "df"]
    )
    gl_error = int(tabla_anova.loc["error", "df"])

    graficar_efectos_principales(
        df,
        niveles,
        mse,
        gl_error,
        DIR_GRAFICAS / "efectos_principales.png",
    )
    graficar_interaccion(
        df, niveles, mse, gl_error, DIR_GRAFICAS / "interaccion.png"
    )
    graficar_boxplots(df, niveles, DIR_GRAFICAS / "boxplots.png")
    graficar_bootstrap(
        bootstrap_df, niveles, DIR_GRAFICAS / "bootstrap_medianas.png"
    )
    graficar_residuos(modelo, DIR_GRAFICAS / "diagnostico_residuos.png")
    graficar_instancias(df, instancias, DIR_GRAFICAS / "instancias.png")
    graficar_calibracion(calibracion, DIR_GRAFICAS / "calibracion.png")

    print("\n" + "=" * 72)
    print("ANOVA DE DOS FACTORES (tipo II)")
    print("=" * 72)
    print(tabla_anova.round(6).to_string())
    print("\nDiagnóstico de supuestos (modelo completo):")
    for clave in ("shapiro_W", "shapiro_p", "levene_F", "levene_p", "r2", "r2_ajustado"):
        print(f"  {clave:<20}: {diagnostico[clave]:.6f}")
    print("\nDiagnóstico de supuestos (sin n = 5):")
    for clave in (
        "shapiro_W_sin_n5",
        "shapiro_p_sin_n5",
        "levene_F_sin_n5",
        "levene_p_sin_n5",
    ):
        print(f"  {clave:<20}: {diagnostico[clave]:.6f}")

    print("\nANOVA DE UN FACTOR POR TAMAÑO")
    print(anova_tamano.round(6).to_string(index=False))
    print("\nKRUSKAL-WALLIS POR TAMAÑO")
    print(kruskal.round(6).to_string(index=False))
    print("\nWELCH ANOVA POR TAMAÑO")
    print(welch.round(6).to_string(index=False))

    print("\nCALIBRACIÓN DEL PRESUPUESTO (5 réplicas por celda)")
    print(resumen_calibracion.round(2).to_string(index=False))

    print("\n" + "=" * 72)
    print("MEJOR TEMPERATURA POR TAMAÑO (mediana bootstrap)")
    print("=" * 72)
    for _, fila in mejor.iterrows():
        empate = " (empate técnico)" if fila["empate"] else ""
        print(
            f"n={int(fila['tamano']):2d}: {fila['nivel_optimo']:<5} "
            f"T={fila['temperatura_optima']:6.2f}  "
            f"mediana={fila['mediana_bootstrap']:7.2f}  "
            f"IC95=[{fila['ic95_inf']:.2f}, {fila['ic95_sup']:.2f}]{empate}"
        )

    print(f"\nResultados guardados en: {RAIZ}")
    return df, tabla_anova, bootstrap_df, mejor


if __name__ == "__main__":
    main()
