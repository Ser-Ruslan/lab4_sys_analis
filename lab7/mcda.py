# ЛР 7. Расчётное ядро: множество Парето, расстояние до идеала, последовательные уступки.
# Здесь только numpy и никакого интерфейса, чтобы расчёты можно было проверить отдельно от Streamlit
# (см. test_mcda.py).
#
# Везде одинаковые входы:
#   values   - матрица n x m, строка = альтернатива, столбец = критерий (в исходных единицах)
#   maximize - m булевых флагов: True если критерий максимизируем, False если минимизируем

from dataclasses import dataclass

import numpy as np

TOL = 1e-9


class McdaError(ValueError):
    """Неправильные входные данные метода (матрица, веса, порядок критериев, уступки)."""


def _check(values, maximize):
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise McdaError("В матрице альтернатив есть нечисловые значения") from exc
    if values.ndim != 2:
        raise McdaError("Матрица альтернатив должна быть двумерной")
    n, m = values.shape
    if n < 2:
        raise McdaError("Нужно минимум две альтернативы")
    if m < 2:
        raise McdaError("Нужно минимум два критерия")
    if not np.isfinite(values).all():
        raise McdaError("В матрице альтернатив есть пропуски или бесконечности")
    maximize = np.asarray(maximize, dtype=bool)
    if maximize.shape != (m,):
        raise McdaError("Число направлений оптимизации не совпадает с числом критериев")
    return values, maximize


def to_max(values, maximize):
    # минимизируемые критерии умножаю на -1, дальше вся задача "на максимум" (как в шаблоне методички)
    values, maximize = _check(values, maximize)
    return np.where(maximize, values, -values)


# Парето

def domination_matrix(values, maximize):
    # dom[a, b] = True, если a доминирует b: не хуже по всем критериям и строго лучше хотя бы по одному
    v = to_max(values, maximize)
    n = len(v)
    dom = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(n):
            if a != b and np.all(v[a] >= v[b]) and np.any(v[a] > v[b]):
                dom[a, b] = True
    return dom


def pareto_mask(values, maximize):
    # альтернатива эффективна, если её никто не доминирует
    return ~domination_matrix(values, maximize).any(axis=0)


# нормирование и расстояние до идеала

def normalize(values, maximize):
    # z = 1 у лучшего значения критерия, z = 0 у худшего; для min-критериев шкала переворачивается сама,
    # потому что они уже со знаком минус
    v = to_max(values, maximize)
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = hi - lo
    z = np.ones_like(v)  # если у всех альтернатив значение одинаковое, критерий ничего не различает -> z = 1
    ok = span > TOL
    z[:, ok] = (v[:, ok] - lo[ok]) / span[ok]
    return z


def ideal_point(values, maximize):
    # идеал в исходных единицах: лучшее значение каждого критерия (в нормированном виде это (1, ..., 1))
    values, maximize = _check(values, maximize)
    return np.where(maximize, values.max(axis=0), values.min(axis=0))


def check_weights(weights, m, tol=1e-6):
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise McdaError("Веса должны быть числами") from exc
    if w.shape != (m,):
        raise McdaError(f"Нужно {m} весов, получено {w.size}")
    if not np.isfinite(w).all() or (w < 0).any():
        raise McdaError("Веса должны быть неотрицательными числами")
    if abs(w.sum() - 1.0) > tol:
        raise McdaError(f"Сумма весов должна быть равна 1, сейчас {w.sum():.4f}")
    return w


def ideal_distance(values, maximize, weights):
    # D(x) = sqrt( sum_i w_i * (1 - z_i(x))^2 ), чем меньше, тем ближе к идеалу
    z = normalize(values, maximize)
    w = check_weights(weights, z.shape[1])
    d = np.sqrt(np.sum(w * (1.0 - z) ** 2, axis=1))
    return z, d


def dense_ranks(scores, ascending=True):
    # места 1, 2, 3... без пропусков; одинаковые значения получают одно место
    s = np.round(np.asarray(scores, dtype=float), 9)  # иначе 0.3 и 0.30000000000000004 считаются разными
    uniq = np.unique(s)
    if not ascending:
        uniq = uniq[::-1]
    return np.array([int(np.where(uniq == x)[0][0]) + 1 for x in s])


# последовательные уступки

@dataclass
class ConcessionStep:
    criterion: int      # индекс критерия, по которому идёт отбор на этом шаге
    best: float         # лучшее значение критерия среди оставшихся
    delta: float        # уступка (на последнем шаге 0)
    threshold: float    # граница: всё, что хуже, отсеивается
    before: list        # кто дошёл до шага
    after: list         # кто остался после шага


@dataclass
class ConcessionResult:
    steps: list
    winners: list            # обычно одна альтернатива, но при равенстве на последнем шаге может быть несколько
    eliminated_at: np.ndarray  # номер шага (с 1), на котором альтернатива выбыла; 0 = не выбыла
    ranks: np.ndarray          # 1 = победитель, дальше чем позже выбыла, тем выше место


def successive_concessions(values, maximize, order, deltas):
    """
    order  - индексы критериев от самого важного к наименее важному;
    deltas - уступки в единицах самого критерия, по одной на каждый критерий кроме последнего.

    На шаге k берём лучшее значение критерия order[k] среди оставшихся альтернатив и оставляем тех,
    кто хуже лучшего не более чем на deltas[k]. На последнем шаге уступки нет: просто лучший.
    """
    values, maximize = _check(values, maximize)
    n, m = values.shape

    order = [int(c) for c in order]
    if not order:
        raise McdaError("Не задан порядок важности критериев")
    if len(set(order)) != len(order):
        raise McdaError("В цепочке приоритетов критерии не должны повторяться")
    if any(c < 0 or c >= m for c in order):
        raise McdaError("В цепочке приоритетов указан несуществующий критерий")

    try:
        deltas = [float(d) for d in deltas]
    except (TypeError, ValueError) as exc:
        raise McdaError("Уступки должны быть числами") from exc
    if len(deltas) != len(order) - 1:
        raise McdaError(f"Нужно {len(order) - 1} уступок (на все критерии кроме последнего), получено {len(deltas)}")
    if any(not np.isfinite(d) or d < 0 for d in deltas):
        raise McdaError("Уступки должны быть неотрицательными числами")

    alive = list(range(n))
    eliminated_at = np.zeros(n, dtype=int)
    steps = []
    for k, c in enumerate(order):
        col = values[alive, c]
        delta = deltas[k] if k < len(order) - 1 else 0.0
        if maximize[c]:
            best = col.max()
            threshold = best - delta
            keep = [i for i in alive if values[i, c] >= threshold - TOL]
        else:
            best = col.min()
            threshold = best + delta
            keep = [i for i in alive if values[i, c] <= threshold + TOL]
        for i in alive:
            if i not in keep:
                eliminated_at[i] = k + 1
        steps.append(ConcessionStep(c, float(best), delta, float(threshold), list(alive), list(keep)))
        alive = keep

    # победителям ставлю "шаг" больше последнего, чтобы они оказались первыми при ранжировании
    survived = np.where(eliminated_at == 0, len(order) + 1, eliminated_at)
    return ConcessionResult(steps, alive, eliminated_at, dense_ranks(survived, ascending=False))
