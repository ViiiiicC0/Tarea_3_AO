# Tarea 3 — Ajuste de temperatura del Recocido Simulado mediante DOE

Implementación del **Ejercicio 1**: ajuste de la temperatura inicial `T0` del
Recocido Simulado con 2-opt sobre el TSP, usando un diseño factorial completo
3×3 con 10 réplicas por punto de diseño (90 corridas). Incluye ANOVA de dos
factores, comparaciones de Tukey, pruebas robustas y bootstrap de la mediana.
El reporte PDF con todas las gráficas se entrega por separado.

## Contenido

| Archivo | Descripción |
|---|---|
| [`src/sa_tsp.py`](src/sa_tsp.py) | SA + 2-opt para TSP, adaptado del Ejercicio 4 de la Tarea 2 con RNG local por corrida. |
| [`src/ejercicio1_doe.py`](src/ejercicio1_doe.py) | Experimento completo: diseño, 90 corridas, ANOVA, Tukey, Kruskal-Wallis, Welch, bootstrap, calibración y gráficas. |
| [`datos/`](datos/) | Resultados crudos y tablas derivadas (CSV/JSON). |
| [`graficas/`](graficas/) | Siete figuras en alta resolución (300 dpi). |

## Resultados destacados

- Matrícula (semilla): **2177628**. Valor único generado:
  `r = 0.5673736344116567`. Rango de temperatura: `[1, 100]`.
- Niveles de temperatura (fracción → `T`):

  | Nivel | Fracción | Temperatura |
  |---|---:|---:|
  | bajo | 0.170212 | 17.85 |
  | medio | 0.470212 | 47.55 |
  | alto | 0.770212 | 77.25 |

- Presupuesto: 500 iteraciones por corrida, enfriamiento exponencial
  `T(t) = T0 · 0.9985^t`. Se calibró este presupuesto porque con 5 000
  iteraciones todas las temperaturas convergen y el factor deja de tener
  efecto.
- ANOVA de dos factores (α = 0.05): temperatura
  (`F(2,81) = 71.20`, `p < 0.001`, `η²p = 0.637`), tamaño
  (`F(2,81) = 1876.36`, `p < 0.001`, `η²p = 0.979`) e interacción
  (`F(4,81) = 38.53`, `p < 0.001`, `η²p = 0.655`). R² = 0.980.

### Temperatura óptima por tamaño

| n | Nivel | T óptima | Mediana bootstrap | IC 95 % |
|---:|---|---:|---:|---|
| 5 | bajo (empate técnico) | 17.85 | 185.08 | [185.08, 185.08] |
| 10 | bajo | 17.85 | 298.47 | [298.47, 300.49] |
| 15 | bajo | 17.85 | 354.96 | [348.72, 360.70] |

En `n = 5` las 30 corridas alcanzan el mismo óptimo (185.08), por lo que los
tres niveles empatan. En `n = 10` y `n = 15` gana claramente el nivel bajo.
La penalización de usar temperatura alta crece con el tamaño (+26.84 y
+102.79 unidades, respectivamente), lo que explica la interacción
significativa. El óptimo está en el borde inferior del rango explorado, por
lo que se recomienda explorar temperaturas aún menores en trabajo futuro.

> Los valores son deterministas: misma matrícula, mismas semillas y mismo
> resultado al repetir los scripts.

## Requisitos

- Python 3.10 o superior (probado con 3.13).
- Dependencias listadas en [`requirements.txt`](requirements.txt): `numpy`,
  `pandas`, `matplotlib`, `scipy`, `statsmodels`, `reportlab`, `pymupdf`.

## Instalación

```bash
git clone https://github.com/ViiiiicC0/Tarea_3_AO.git
cd Tarea_3_AO
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

```bash
# Genera datos/ y graficas/ (90 corridas + calibración)
python src/ejercicio1_doe.py
```

Si se ejecuta en un entorno sin pantalla, el primer script ya usa el backend
`Agg` de Matplotlib.

## Diseño experimental

- **Factor:** temperatura inicial `T0`, tres niveles calculados con un único
  `r ~ U(0,1)` generado con `random.seed(2177628); r = random.random()`.
- **Bloque:** tamaño del TSP `n ∈ {5, 10, 15}`, una instancia fija por tamaño
  (coordenadas uniformes en `[0, 100]²`).
- **Réplicas:** 10 semillas distintas por combinación temperatura × tamaño.
- **Respuesta:** mejor costo histórico del tour (menor es mejor).
- **Análisis:** ANOVA tipo II, Tukey por tamaño, Kruskal-Wallis y Welch como
  pruebas robustas, y mediana bootstrap con 10 000 remuestreos e IC
  percentil del 95 %.

La semilla de cada corrida se calcula como
`2177628 + 1000*n + 100*i + replica`, con `i = 0, 1, 2` para los niveles
bajo, medio y alto. El valor `r` se genera una sola vez y se reutiliza en
todo el experimento.

## Estructura del repositorio

```
Tarea_3_AO/
├── README.md
├── requirements.txt
├── src/
│   ├── sa_tsp.py
│   └── ejercicio1_doe.py
├── datos/
│   ├── resultados.csv
│   ├── diseno.json
│   ├── instancias.json
│   ├── anova.csv
│   ├── anova_por_tamano.csv
│   ├── tukey_por_tamano.csv
│   ├── kruskal_por_tamano.csv
│   ├── welch_por_tamano.csv
│   ├── anova_sin_n5.csv
│   ├── bootstrap_puntos.csv
│   ├── mejor_por_tamano.csv
│   ├── calibracion.csv
│   └── calibracion_resumen.csv
└── graficas/
    ├── instancias.png
    ├── calibracion.png
    ├── efectos_principales.png
    ├── interaccion.png
    ├── boxplots.png
    ├── diagnostico_residuos.png
    └── bootstrap_medianas.png
```

## Gráficas

| Gráfica | Archivo |
|---|---|
| Instancias y mejor tour | [`instancias.png`](graficas/instancias.png) |
| Calibración del presupuesto | [`calibracion.png`](graficas/calibracion.png) |
| Efectos principales | [`efectos_principales.png`](graficas/efectos_principales.png) |
| Interacción temperatura × tamaño | [`interaccion.png`](graficas/interaccion.png) |
| Distribuciones por celda | [`boxplots.png`](graficas/boxplots.png) |
| Diagnóstico de residuos | [`diagnostico_residuos.png`](graficas/diagnostico_residuos.png) |
| Medianas bootstrap e IC 95 % | [`bootstrap_medianas.png`](graficas/bootstrap_medianas.png) |

## Reporte

El análisis completo, las tablas ANOVA y bootstrap, las gráficas y las
respuestas a las preguntas analíticas se entregan en un PDF por separado;
ese archivo no forma parte de este repositorio.

**Repositorio público:** https://github.com/ViiiiicC0/Tarea_3_AO
