"""Лабораторная 5, вариант 1: непрерывная модель производственного плана."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

DEFAULT_A = [[1, 2, 1], [2, 1, 3], [4, 2, 1]]
DEFAULT_B = [420, 600, 900]
DEFAULT_C = [3, 3, 4]


@dataclass
class Result:
    status: str
    message: str
    x: object = None
    value: float = 0.0
    used: object = None
    slack: object = None
    shadow: object = None
    binding: object = None


def validate(A, b, c):
    A, b, c = (np.asarray(v, dtype=float) for v in (A, b, c))
    if A.ndim != 2 or b.ndim != 1 or c.ndim != 1 or A.shape != (len(b), len(c)):
        raise ValueError('Размеры матрицы A и векторов b, c не согласованы.')
    if not len(b) or not len(c):
        raise ValueError('Задача должна содержать ресурсы и продукцию.')
    if any(not np.isfinite(v).all() for v in (A, b, c)):
        raise ValueError('Введите конечные числа, без NaN и бесконечности.')
    if any((v < 0).any() for v in (A, b, c)):
        raise ValueError('Расход, запасы и стоимость не могут быть отрицательными.')
    return A, b, c


def solve(A=DEFAULT_A, b=DEFAULT_B, c=DEFAULT_C):
    A, b, c = validate(A, b, c)
    # HiGHS dual simplex. linprog минимизирует, поэтому меняем знак c.
    raw = linprog(-c, A_ub=A, b_ub=b, bounds=(0, None), method='highs-ds')
    statuses = {0: ('OPTIMAL', 'Решение успешно найдено'),
                1: ('LIMIT', 'Достигнут лимит вычислений'),
                2: ('INFEASIBLE', 'Нет допустимых решений'),
                3: ('UNBOUNDED', 'Целевая функция не ограничена'),
                4: ('ERROR', 'Численная ошибка солвера')}
    status, message = statuses.get(raw.status, ('ERROR', raw.message))
    if not raw.success:
        return Result(status, message)
    used = A @ raw.x
    slack = np.maximum(0.0, b - used)
    shadow = -raw.ineqlin.marginals
    tol = 1e-7 * np.maximum(1.0, np.abs(b))
    return Result(status, message, raw.x, float(c @ raw.x), used, slack,
                  shadow, slack <= tol)


def sensitivity(A, b, c):
    """Меняем запас одного ресурса, остальные параметры фиксируем."""
    A, b, c = validate(A, b, c)
    base = solve(A, b, c)
    if base.status != 'OPTIMAL':
        return []
    rows = []
    for i in range(len(b)):
        for delta in (-0.1 * b[i], 1.0, 0.1 * b[i]):
            changed = b.copy()
            changed[i] += delta
            result = solve(A, changed, c)
            rows.append({'resource': i + 1, 'delta': float(delta),
                         'stock': float(changed[i]), 'status': result.status,
                         'value': result.value if result.status == 'OPTIMAL' else None,
                         'gain': result.value - base.value if result.status == 'OPTIMAL' else None})
    return rows


if __name__ == '__main__':
    result = solve()
    print(result)
    for row in sensitivity(DEFAULT_A, DEFAULT_B, DEFAULT_C):
        print(row)
