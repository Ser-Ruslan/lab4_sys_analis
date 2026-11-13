# ЛР 7, вариант 1. Многокритериальный выбор: Парето, расстояние до идеала, последовательные уступки.
# Запуск:  streamlit run lab7/app.py
#
# Все параметры (данные, направления, веса, порядок критериев, уступки) задаются в интерфейсе,
# код для другого варианта менять не надо: достаточно выбрать другой файл.

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
import data_io
import mcda

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_FILE = "variant_01.csv"
MAX_LABEL, MIN_LABEL = "↑ максимум", "↓ минимум"


# ---------- источник данных ----------

def choose_source():
    """Возвращает (таблица, ключ источника). Ключ нужен, чтобы виджеты сбрасывались при смене данных."""
    st.sidebar.header("1. Данные")
    mode = st.sidebar.radio("Откуда взять альтернативы", ["Готовый вариант", "Файл CSV / Excel", "Ручной ввод"])

    if mode == "Готовый вариант":
        files = sorted(p.name for p in DATA_DIR.glob("*.csv"))
        if not files:
            raise data_io.DataError(f"В папке {DATA_DIR} нет CSV-файлов, загрузите таблицу сами")
        index = files.index(DEFAULT_FILE) if DEFAULT_FILE in files else 0
        name = st.sidebar.selectbox("Файл варианта", files, index=index)
        return data_io.read_table(DATA_DIR / name), f"file:{name}"

    if mode == "Файл CSV / Excel":
        up = st.sidebar.file_uploader("Таблица альтернатив", type=["csv", "xlsx"])
        st.sidebar.caption("Первый столбец — названия, остальные — критерии. "
                           "Суффикс _max / _min в заголовке задаёт направление.")
        if up is None:
            st.info("Загрузите CSV или XLSX в боковой панели.")
            st.stop()
        return data_io.read_table(up, up.name), f"upload:{up.name}:{up.size}"

    raw = st.sidebar.text_input("Критерии через запятую", "f1_Accuracy_max, f2_Latency_min, f3_Cost_min")
    criteria = [c.strip() for c in raw.split(",") if c.strip()]
    if len(criteria) < 2:
        raise data_io.DataError("Для ручного ввода нужно минимум два критерия")
    if len(set(criteria)) != len(criteria):
        raise data_io.DataError("Названия критериев повторяются")
    return data_io.empty_table(criteria), "manual:" + "|".join(criteria)


def edit_problem(df, source_key):
    """Таблица с правкой вручную + выбор направления по каждому критерию."""
    st.subheader("Исходные данные")
    st.caption("Таблицу можно править прямо здесь: менять значения, добавлять и удалять строки.")
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", hide_index=True, key=f"editor:{source_key}")

    columns = list(df.columns[1:])
    maximize = []
    for box, col in zip(st.columns(len(columns)), columns):
        clean, flag = data_io.parse_direction(col)
        choice = box.selectbox(f"{clean}", [MAX_LABEL, MIN_LABEL], index=0 if flag in (True, None) else 1,
                               key=f"dir:{source_key}:{col}")
        maximize.append(choice == MAX_LABEL)
    return data_io.build_problem(edited, maximize)


# ---------- параметры методов ----------

def default_weights(m):
    if m == 3:
        return [0.4, 0.3, 0.3]  # как в примере из методички
    w = [round(1.0 / m, 2)] * m
    w[0] = round(1.0 - sum(w[1:]), 2)
    return w


def normalize_weights(keys):
    # колбэк кнопки: пересчитывает слайдеры так, чтобы сумма стала ровно 1 (с шагом 0.01)
    w = np.array([st.session_state[k] for k in keys], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(keys))
    w = np.round(w / w.sum(), 2)
    w[np.argmax(w)] = round(w[np.argmax(w)] + 1.0 - w.sum(), 2)  # остаток от округления отдаю самому большому весу
    for k, x in zip(keys, w):
        st.session_state[k] = float(min(max(x, 0.0), 1.0))


def weight_controls(problem, key):
    st.sidebar.header("2. Веса критериев ωᵢ")
    keys = [f"w:{key}:{i}" for i in range(len(problem.criteria))]
    for k, w0 in zip(keys, default_weights(len(keys))):
        st.session_state.setdefault(k, w0)
    weights = [st.sidebar.slider(c, 0.0, 1.0, step=0.01, key=k) for c, k in zip(problem.criteria, keys)]

    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        st.sidebar.error(f"Сумма весов = {total:.2f}, а должна быть 1. Метод идеала не считается.")
        st.sidebar.button("Нормировать веса до 1", on_click=normalize_weights, args=(keys,))
        return None
    st.sidebar.success("Сумма весов = 1.00")
    return weights


def concession_controls(problem, key):
    st.sidebar.header("3. Метод уступок")
    order_names = st.sidebar.multiselect("Критерии от самого важного к наименее важному", problem.criteria,
                                         default=problem.criteria, key=f"order:{key}")
    if len(order_names) != len(problem.criteria):
        st.sidebar.error("В цепочке приоритетов должны быть все критерии.")
        return None, None
    order = [problem.criteria.index(c) for c in order_names]

    deltas = []
    for c in order[:-1]:
        col = problem.values[:, c]
        default = round(0.3 * float(col.max() - col.min()), 4)  # по умолчанию 30% размаха критерия
        deltas.append(st.sidebar.number_input(f"Уступка Δ по «{problem.criteria[c]}»", min_value=0.0, value=default,
                                              format="%g", key=f"delta:{key}:{c}",
                                              help="В единицах самого критерия: насколько можно быть хуже лучшего."))
    st.sidebar.caption(f"По последнему критерию («{order_names[-1]}») уступки нет: выбирается лучший из оставшихся.")
    return order, deltas


# ---------- вывод результатов ----------

def fmt_names(problem, idx):
    return ", ".join(problem.names[i] for i in idx) if len(idx) else "—"


def space_coords(problem, z, normalized):
    # в каком пространстве рисовать: исходные единицы или нормированные z (там идеал = (1, ..., 1))
    if normalized:
        return z, np.ones(z.shape[1]), [f"z: {c}" for c in problem.criteria]
    arrows = ["↑" if f else "↓" for f in problem.maximize]
    return (problem.values, mcda.ideal_point(problem.values, problem.maximize),
            [f"{c} {a}" for c, a in zip(problem.criteria, arrows)])


def pareto_tab(problem, z, pareto, dom, marks, key):
    n_eff = int(pareto.sum())
    st.markdown(f"**Парето-эффективных альтернатив: {n_eff} из {len(problem.names)}** — "
                f"{fmt_names(problem, np.where(pareto)[0])}")

    normalized = st.radio("Пространство", ["Исходные единицы", "Нормированные критерии z"], horizontal=True,
                          key=f"space:{key}") == "Нормированные критерии z"
    coords, ideal, titles = space_coords(problem, z, normalized)
    hover = charts.hover_text(problem.names, problem.values, problem.criteria)
    m = len(problem.criteria)

    if m >= 3:
        axes = [0, 1, 2]
        if m > 3:
            picked = st.multiselect("Три критерия для осей 3D", problem.criteria, default=problem.criteria[:3],
                                    max_selections=3, key=f"axes3:{key}")
            axes = [problem.criteria.index(c) for c in picked]
        if len(axes) == 3:
            st.plotly_chart(charts.alternatives_scatter(problem.names, coords[:, axes], [titles[a] for a in axes],
                                                        pareto, ideal[axes], hover, marks), width="stretch")
        else:
            st.warning("Для 3D-графика выберите ровно три критерия.")

    st.markdown("**Проекция на два критерия**")
    left, right = st.columns(2)
    ax = left.selectbox("Ось X", problem.criteria, index=0, key=f"ax:{key}")
    ay = right.selectbox("Ось Y", problem.criteria, index=1, key=f"ay:{key}")
    if ax == ay:
        st.warning("Для проекции выберите два разных критерия.")
    else:
        pair = [problem.criteria.index(ax), problem.criteria.index(ay)]
        front = charts.projection_front(problem.values, problem.maximize, pair)
        st.plotly_chart(charts.alternatives_scatter(problem.names, coords[:, pair], [titles[a] for a in pair],
                                                    pareto, ideal[pair], hover, marks, front), width="stretch")
        if m > 2:
            st.caption("Цвет точки — эффективность по всем критериям сразу; пунктир — граница Парето, "
                       "если смотреть только на эти два критерия.")

    table = pd.DataFrame({
        "Альтернатива": problem.names,
        "Статус": ["Парето-эффективна" if p else "Доминируется" for p in pareto],
        "Кем доминируется": [fmt_names(problem, np.where(dom[:, i])[0]) for i in range(len(problem.names))],
    })
    st.dataframe(table, hide_index=True, width="stretch")


def ideal_tab(problem, z, weights, distances, pareto):
    st.latex(r"D(x)=\sqrt{\sum_{i=1}^{m}\omega_i\,(1-z_i(x))^2},\qquad Z^*=(1,\dots,1)")
    if distances is None:
        st.warning("Сумма весов не равна 1 — исправьте веса в боковой панели.")
        return
    table = pd.DataFrame(z, columns=[f"z: {c}" for c in problem.criteria])
    table.insert(0, "Альтернатива", problem.names)
    table["D(x)"] = distances
    table["Место"] = mcda.dense_ranks(distances)
    table = table.sort_values(["Место", "Альтернатива"])
    st.dataframe(table.style.format(precision=4), hide_index=True, width="stretch")
    st.caption("Веса: " + ", ".join(f"{c} = {w:.2f}" for c, w in zip(problem.criteria, weights)))
    st.plotly_chart(charts.distance_bars(problem.names, distances, pareto), width="stretch")
    best = np.where(mcda.dense_ranks(distances) == 1)[0]
    st.success(f"Ближе всех к идеалу: **{fmt_names(problem, best)}** (D = {distances[best[0]]:.4f})")


def concession_tab(problem, result):
    if result is None:
        st.warning("Задайте полную цепочку приоритетов в боковой панели.")
        return
    rows = []
    for k, s in enumerate(result.steps, start=1):
        sign = "≥" if problem.maximize[s.criterion] else "≤"
        last = k == len(result.steps)
        rows.append({
            "Шаг": k,
            "Критерий": problem.criteria[s.criterion],
            "Лучшее значение": s.best,
            "Уступка Δ": "—" if last else f"{s.delta:g}",
            "Условие отбора": f"{sign} {s.threshold:g}",
            "Остались": fmt_names(problem, s.after),
            "Выбыли": fmt_names(problem, [i for i in s.before if i not in s.after]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.success(f"Решение методом уступок: **{fmt_names(problem, result.winners)}**")


def summary_tab(problem, pareto, distances, result):
    table = pd.DataFrame(problem.values, columns=problem.criteria)
    table.insert(0, "Альтернатива", problem.names)
    table["Парето"] = ["да" if p else "нет" for p in pareto]
    sort_by = ["Альтернатива"]
    if result is not None:
        table["Место (уступки)"] = result.ranks
        table["Выбыла на шаге"] = [str(s) if s else "—" for s in result.eliminated_at]
        sort_by.insert(0, "Место (уступки)")
    if distances is not None:
        table["D(x)"] = np.round(distances, 4)
        table["Место (идеал)"] = mcda.dense_ranks(distances)
        sort_by.insert(0, "Место (идеал)")
    table = table.sort_values(sort_by)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button("Скачать таблицу (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
                       file_name="lab7_result.csv", mime="text/csv")

    best_ideal = set(np.where(mcda.dense_ranks(distances) == 1)[0]) if distances is not None else set()
    best_conc = set(result.winners) if result is not None else set()
    if best_ideal and best_conc:
        common = sorted(best_ideal & best_conc)
        if common:
            eff = "входит в множество Парето" if all(pareto[i] for i in common) else "НЕ входит в множество Парето"
            st.success(f"Оба метода выбирают **{fmt_names(problem, common)}**; альтернатива {eff}.")
        else:
            st.info(f"Методы расходятся: по расстоянию до идеала — **{fmt_names(problem, sorted(best_ideal))}**, "
                    f"по уступкам — **{fmt_names(problem, sorted(best_conc))}**. Метод идеала ищет компромисс "
                    "по всем критериям сразу, а метод уступок жёстко следует порядку важности.")


def main():
    st.set_page_config(page_title="ЛР 7 — многокритериальный выбор", layout="wide")
    st.title("ЛР 7. Многокритериальный анализ альтернатив")

    try:
        df, source_key = choose_source()
        problem = edit_problem(df, source_key)
    except (data_io.DataError, mcda.McdaError) as exc:
        st.error(f"Ошибка в данных: {exc}")
        st.stop()

    # при изменении набора критериев ключи виджетов меняются, и веса/уступки сбрасываются на значения по умолчанию
    key = source_key + "#" + "|".join(problem.criteria)
    weights = weight_controls(problem, key)
    order, deltas = concession_controls(problem, key)

    try:
        dom = mcda.domination_matrix(problem.values, problem.maximize)
        pareto = ~dom.any(axis=0)
        z = mcda.normalize(problem.values, problem.maximize)
        distances = mcda.ideal_distance(problem.values, problem.maximize, weights)[1] if weights else None
        result = (mcda.successive_concessions(problem.values, problem.maximize, order, deltas)
                  if order is not None else None)
    except mcda.McdaError as exc:
        st.error(f"Ошибка в параметрах: {exc}")
        st.stop()

    marks = {}
    if distances is not None:
        marks["Выбор: расстояние до идеала"] = ([int(i) for i in np.where(mcda.dense_ranks(distances) == 1)[0]],
                                                charts.COLOR_BEST)
    if result is not None:
        marks["Выбор: метод уступок"] = (list(result.winners), charts.COLOR_CONCESSION)

    tabs = st.tabs(["Множество Парето", "Расстояние до идеала", "Метод уступок", "Итоговая таблица"])
    with tabs[0]:
        pareto_tab(problem, z, pareto, dom, marks, key)
    with tabs[1]:
        ideal_tab(problem, z, weights, distances, pareto)
    with tabs[2]:
        concession_tab(problem, result)
    with tabs[3]:
        summary_tab(problem, pareto, distances, result)


main()
