"""Núcleo del Recocido Simulado con 2-opt para el TSP.

Reutiliza la lógica del Ejercicio 4 de la Tarea 2 (2-opt aleatorio,
criterio de Metropolis y enfriamiento exponencial), pero recibe un
generador aleatorio local para que cada corrida sea reproducible de
forma independiente del estado global de ``random``.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

Point = tuple[float, float]
Route = list[int]


@dataclass
class SAResult:
    """Resultado de una corrida de Recocido Simulado."""

    best_route: Route
    best_cost: float
    iterations: int
    elapsed_ms: float


def tour_length(points: list[Point], route: Route) -> float:
    """Calcula la distancia euclidiana total de un tour cerrado."""
    n = len(route)
    return sum(
        math.dist(points[route[i]], points[route[(i + 1) % n]]) for i in range(n)
    )


def two_opt_move(route: Route, i: int, k: int) -> Route:
    """Invierte el subsegmento ``route[i..k]`` (movimiento 2-opt)."""
    return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]


def generate_instance(n: int, seed: int) -> list[Point]:
    """Genera una instancia euclidiana con ``n`` ciudades en [0, 100]^2."""
    rng = random.Random(seed)
    return [(rng.uniform(0.0, 100.0), rng.uniform(0.0, 100.0)) for _ in range(n)]


def random_initial_route(n: int, rng: random.Random) -> Route:
    """Genera una permutación inicial aleatoria de las ciudades."""
    route = list(range(n))
    rng.shuffle(route)
    return route


def simulated_annealing(
    points: list[Point],
    initial_route: Route,
    T0: float,
    max_iter: int = 5000,
    cooling: float = 0.9985,
    rng: random.Random | None = None,
) -> SAResult:
    """Ejecuta SA con 2-opt aleatorio y enfriamiento exponencial.

    Parameters
    ----------
    points:
        Coordenadas de las ciudades.
    initial_route:
        Tour de arranque (permutación de ``range(n)``).
    T0:
        Temperatura inicial del esquema exponencial ``T(t) = T0 * cooling**t``.
    max_iter:
        Número máximo de iteraciones.
    cooling:
        Factor de enfriamiento multiplicativo.
    rng:
        Generador aleatorio local. Si es ``None`` se usa ``random`` global.
    """
    if rng is None:
        rng = random

    t_start = time.perf_counter()
    n = len(initial_route)

    current_route = initial_route[:]
    current_cost = tour_length(points, current_route)

    best_route = current_route[:]
    best_cost = current_cost
    iterations = 0

    for t in range(1, max_iter + 1):
        iterations = t
        T = T0 * (cooling**t)
        if T <= 1e-4:
            break

        # Vecino 2-opt aleatorio: dos índices de corte i < k.
        i, k = sorted(rng.sample(range(1, n), 2))
        candidate_route = two_opt_move(current_route, i, k)
        candidate_cost = tour_length(points, candidate_route)

        delta = candidate_cost - current_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_route = candidate_route
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_route = current_route[:]
                best_cost = current_cost

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return SAResult(best_route, best_cost, iterations, elapsed_ms)
