"""Модель Уилсона. Расчёты без округления промежуточных результатов."""
from dataclasses import dataclass, asdict
import math


VARIANTS = {
    1: (8000, 250, 5.0, 4, 120), 2: (15000, 400, 8.5, 6, 210),
    3: (5400, 180, 3.2, 3, 90), 4: (22000, 600, 12.0, 7, 350),
    5: (11000, 310, 6.0, 5, 180), 6: (9500, 280, 4.8, 4, 140),
    7: (17500, 450, 9.0, 8, 260), 8: (6200, 210, 3.8, 3, 100),
    9: (13000, 350, 7.2, 5, 195), 10: (19000, 520, 10.5, 6, 310),
    11: (7300, 230, 4.2, 4, 115), 12: (14200, 380, 8.0, 5, 225),
    13: (25000, 700, 15.0, 10, 400),
}


@dataclass(frozen=True)
class Parameters:
    D: float = 8000
    Co: float = 250
    Ch: float = 5
    L: float = 4
    Cp: float = 120

    def validate(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name}: нужно конечное число.")
            if value < 0 or (name in ('D', 'Co', 'Ch') and value == 0):
                condition = 'больше нуля' if name in ('D', 'Co', 'Ch') else 'не меньше нуля'
                raise ValueError(f"{name}: значение должно быть {condition}.")


@dataclass(frozen=True)
class Result:
    Q: float
    N: float
    T: float
    daily_demand: float
    ROP: float
    ordering: float
    holding: float
    purchase: float
    management: float
    total: float
    integer_Q: int
    integer_total: float


def costs(p, quantity):
    p.validate()
    if not math.isfinite(quantity) or quantity <= 0:
        raise ValueError('Размер партии должен быть конечным числом больше нуля.')
    ordering = p.D / quantity * p.Co
    holding = quantity / 2 * p.Ch
    purchase = p.D * p.Cp
    return ordering, holding, purchase, ordering + holding + purchase


def calculate(p):
    p.validate()
    q = math.sqrt(2 * p.D * p.Co / p.Ch)
    ordering, holding, purchase, total = costs(p, q)
    candidates = {max(1, math.floor(q)), max(1, math.ceil(q))}
    integer_q = min(candidates, key=lambda x: costs(p, x)[3])
    return Result(q, p.D / q, 365 * q / p.D, p.D / 365,
                  p.D / 365 * p.L, ordering, holding, purchase,
                  ordering + holding, total, integer_q, costs(p, integer_q)[3])


def inventory_points(p, result, cycles=3):
    """Точные границы циклов, включая вертикальные скачки поставки."""
    points = [(0.0, result.Q)]
    for cycle in range(1, cycles + 1):
        time = cycle * result.T
        points.extend([(time, 0.0), (time, result.Q)])
    return points


def schedule(p, result, cycles=3):
    """Отрицательный день заказа означает заказ до начала наблюдения."""
    return [{'cycle': i, 'order_day': i * result.T - p.L,
             'arrival_day': i * result.T, 'quantity': result.Q}
            for i in range(1, cycles + 1)]
