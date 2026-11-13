# ЛР 7. Загрузка таблицы альтернатив (CSV / Excel) и проверка, что с ней можно считать.
#
# Формат таблицы: первый столбец - названия альтернатив, остальные - критерии.
# Направление оптимизации берётся из названия столбца: суффикс _max / _min (как в файлах из Lb_7.zip)
# или стрелка ↑ / ↓. Если ничего нет, направление выбирается в интерфейсе.

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """Таблицу альтернатив не удалось прочитать или в ней ошибка."""


@dataclass
class Problem:
    names: list            # названия альтернатив
    criteria: list         # названия критериев
    values: np.ndarray     # n x m
    maximize: np.ndarray   # m флагов


def parse_direction(column):
    # 'f2_Latency_min' -> ('f2_Latency', False); 'Точность ↑' -> ('Точность', True); 'Цена' -> ('Цена', None)
    name = str(column).strip()
    low = name.lower()
    for suffix, flag in (("_max", True), ("_min", False)):
        if low.endswith(suffix):
            return name[: -len(suffix)], flag
    for arrow, flag in (("↑", True), ("↓", False)):
        if arrow in name:
            return name.replace(arrow, "").strip(), flag
    return name, None


def read_table(source, filename=None):
    """source - путь к файлу или байты/файловый объект (тогда нужен filename, чтобы понять формат)."""
    name = str(filename or source)
    ext = Path(name).suffix.lower()
    try:
        if ext == ".xlsx":
            df = pd.read_excel(source)
        elif ext in (".csv", ".txt"):
            raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()
            df = _read_csv_bytes(raw)
        else:
            raise DataError(f"Неизвестный формат файла «{ext or name}», нужен .csv или .xlsx")
    except DataError:
        raise
    except FileNotFoundError as exc:
        raise DataError(f"Файл не найден: {name}") from exc
    except Exception as exc:  # pandas кидает очень разные исключения на битых файлах
        raise DataError(f"Не получилось прочитать файл {Path(name).name}: {exc}") from exc

    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < 3:
        raise DataError("В таблице должен быть столбец с названиями альтернатив и минимум два критерия")
    return df.reset_index(drop=True)


def _read_csv_bytes(raw):
    # Excel по-русски сохраняет CSV в cp1251 и с ';', поэтому пробую обе кодировки и даю pandas угадать разделитель
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DataError("Не удалось определить кодировку CSV (ожидается UTF-8 или Windows-1251)")
    if not text.strip():
        raise DataError("Файл пустой")
    return pd.read_csv(io.StringIO(text), sep=None, engine="python")


def empty_table(criteria, rows=4):
    # заготовка для ручного ввода
    data = {"Альтернатива": [f"A{i + 1}" for i in range(rows)]}
    for c in criteria:
        data[c] = [None] * rows
    df = pd.DataFrame(data)
    for c in criteria:
        df[c] = df[c].astype(float)
    return df


def build_problem(df, maximize):
    """Из таблицы (первый столбец - названия) и флагов направлений собирает задачу, попутно всё проверяя."""
    df = df.dropna(how="all")
    if df.shape[1] < 3:
        raise DataError("Нужен столбец с названиями альтернатив и минимум два критерия")
    if len(df) < 2:
        raise DataError("Нужно минимум две альтернативы")

    names = df.iloc[:, 0]
    if names.isna().any() or (names.astype(str).str.strip() == "").any():
        raise DataError("У каждой альтернативы должно быть название (первый столбец)")
    names = [str(x).strip() for x in names]
    dupes = sorted({x for x in names if names.count(x) > 1})
    if dupes:
        raise DataError("Названия альтернатив повторяются: " + ", ".join(dupes))

    criteria = [parse_direction(c)[0] for c in df.columns[1:]]
    if len(set(criteria)) != len(criteria):
        raise DataError("Названия критериев повторяются")
    if len(maximize) != len(criteria):
        raise DataError("Число направлений оптимизации не совпадает с числом критериев")

    body = df.iloc[:, 1:]
    # запятая как десятичный разделитель тоже допустима ("0,82")
    numeric = body.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce")
                         if col.dtype == object else pd.to_numeric(col, errors="coerce"))
    bad = numeric.isna() | ~np.isfinite(numeric.fillna(0.0))
    if bad.any().any():
        row, col = np.argwhere(bad.to_numpy())[0]
        raise DataError(f"Пустое или нечисловое значение: альтернатива «{names[row]}», критерий «{criteria[col]}»")

    return Problem(names, criteria, numeric.to_numpy(dtype=float), np.asarray(maximize, dtype=bool))
