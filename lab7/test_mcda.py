# Проверка ядра на варианте 1 и на паре краевых случаев. Запуск: python lab7/test_mcda.py
# Ожидаемые числа посчитаны руками (см. отчёт).

from pathlib import Path

import numpy as np

import data_io
import mcda

NAMES = ["Model_Alpha", "Model_Beta", "Model_Gamma", "Model_Delta", "Model_Epsilon"]
X = np.array([[85, 45, 120], [82, 30, 150], [78, 20, 90], [88, 60, 200], [80, 25, 110]], dtype=float)
MAXIMIZE = [True, False, False]


def test_csv_variant1():
    df = data_io.read_table(Path(__file__).parent / "data" / "variant_01.csv")
    flags = [data_io.parse_direction(c)[1] for c in df.columns[1:]]
    p = data_io.build_problem(df, flags)
    assert p.names == NAMES
    assert list(p.maximize) == MAXIMIZE
    assert np.array_equal(p.values, X)


def test_pareto_variant1():
    # Beta доминируется моделью Epsilon? нет: у Epsilon точность 80 < 82. В варианте 1 эффективны все пять
    assert mcda.pareto_mask(X, MAXIMIZE).all()


def test_pareto_dominated():
    # добавляю заведомо плохую модель: хуже Alpha по всем трём критериям
    X2 = np.vstack([X, [84, 50, 130]])
    mask = mcda.pareto_mask(X2, MAXIMIZE)
    assert list(mask) == [True] * 5 + [False]
    assert mcda.domination_matrix(X2, MAXIMIZE)[0, 5]
    # одинаковые альтернативы друг друга не доминируют
    assert mcda.pareto_mask(np.array([[1, 1], [1, 1]]), [True, True]).all()


def test_normalize():
    z = mcda.normalize(X, MAXIMIZE)
    assert np.allclose(z[0], [0.7, 0.375, 80 / 110])      # Alpha
    assert np.allclose(z[2], [0.0, 1.0, 1.0])             # Gamma: худшая точность, лучшие задержка и цена
    assert np.allclose(z[3], [1.0, 0.0, 0.0])             # Delta наоборот
    # критерий с одинаковыми значениями не должен давать деление на ноль
    assert np.allclose(mcda.normalize(np.array([[1, 5], [2, 5]]), [True, True])[:, 1], 1.0)


def test_ideal_distance():
    _, d = mcda.ideal_distance(X, MAXIMIZE, [0.4, 0.3, 0.3])
    assert np.allclose(d, [0.4189, 0.5020, 0.6325, 0.7746, 0.5202], atol=1e-4)
    assert NAMES[int(np.argmin(d))] == "Model_Alpha"
    assert list(mcda.dense_ranks(d)) == [1, 2, 4, 5, 3]


def test_weights_validation():
    for bad in ([0.5, 0.3, 0.3], [0.5, 0.5], [1.2, -0.1, -0.1]):
        try:
            mcda.ideal_distance(X, MAXIMIZE, bad)
        except mcda.McdaError:
            continue
        raise AssertionError(f"веса {bad} должны были не пройти проверку")


def test_concessions():
    # точность -> задержка -> цена, уступки 3 % и 12 мс
    r = mcda.successive_concessions(X, MAXIMIZE, [0, 1, 2], [3, 12])
    assert r.steps[0].after == [0, 3]       # точность >= 85: Alpha, Delta
    assert r.steps[1].after == [0]          # задержка <= 57: Alpha
    assert [NAMES[i] for i in r.winners] == ["Model_Alpha"]
    assert list(r.eliminated_at) == [0, 1, 1, 2, 1]
    assert list(r.ranks) == [1, 3, 3, 2, 3]

    # без уступок это лексикографический выбор: побеждает самая точная
    r0 = mcda.successive_concessions(X, MAXIMIZE, [0, 1, 2], [0, 0])
    assert [NAMES[i] for i in r0.winners] == ["Model_Delta"]

    # другой порядок: сначала цена
    r2 = mcda.successive_concessions(X, MAXIMIZE, [2, 1, 0], [30, 5])
    assert [NAMES[i] for i in r2.winners] == ["Model_Epsilon"]

    for order, deltas in (([0, 0, 1], [1, 1]), ([0, 1, 2], [1]), ([0, 1, 2], [1, -1]), ([0, 5], [1])):
        try:
            mcda.successive_concessions(X, MAXIMIZE, order, deltas)
        except mcda.McdaError:
            continue
        raise AssertionError(f"order={order}, deltas={deltas} должны были не пройти проверку")


def test_bad_table():
    import pandas as pd
    for df in (pd.DataFrame({"a": ["x", "x"], "f1": [1, 2], "f2": [3, 4]}),        # повтор названий
               pd.DataFrame({"a": ["x", "y"], "f1": [1, "abc"], "f2": [3, 4]}),    # текст вместо числа
               pd.DataFrame({"a": ["x", "y"], "f1": [1, None], "f2": [3, 4]}),     # пропуск
               pd.DataFrame({"a": ["x"], "f1": [1], "f2": [3]})):                  # одна альтернатива
        try:
            data_io.build_problem(df, [True, True])
        except data_io.DataError:
            continue
        raise AssertionError("таблица с ошибкой должна была не пройти проверку")
    # десятичная запятая допустима
    p = data_io.build_problem(pd.DataFrame({"a": ["x", "y"], "f1": ["0,5", "1,5"], "f2": [3, 4]}), [True, False])
    assert np.allclose(p.values[:, 0], [0.5, 1.5])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"Все проверки пройдены: {len(tests)}")
