# ЛР 6, вариант 1. Критерии Вальда, максимакса, Сэвиджа и Гурвица + окно на tkinter.
# Запуск: python lab6_decision.py (нужны numpy и matplotlib)
#
# Гурвиц считается как в методичке: H = a*min + (1-a)*max,
# то есть a=1 это Вальд, a=0 это максимакс. В некоторых учебниках наоборот, не путать.

import numpy as np

# Вариант №1
DEFAULT_MATRIX = np.array([
    [45, 30, 15, -5],
    [20, 55, 35, 10],
    [10, 25, 70, 40],
    [5, 15, 30, 85],
], dtype=float)

ALPHAS = np.round(np.arange(0.0, 1.0001, 0.1), 1)  # 0, 0.1 ... 1.0 (round, иначе вылезает 0.30000000000000004)


# расчёты

def best_indices(values, maximize=True, tol=1e-9):
    # возвращаю все лучшие, а не только первую: у Вальда в моём варианте A2 и A3 дают по 10,
    # а np.argmax вернул бы только A2
    values = np.asarray(values, dtype=float)
    target = values.max() if maximize else values.min()
    return [i for i, v in enumerate(values) if abs(v - target) < tol]


def wald(X):
    mins = X.min(axis=1)
    return mins, best_indices(mins, maximize=True)


def maximax(X):
    maxs = X.max(axis=1)
    return maxs, best_indices(maxs, maximize=True)


def risk_matrix(X):
    # r_ij = максимум по столбцу минус элемент (numpy сам растянет строку максимумов на всю матрицу)
    return X.max(axis=0) - X


def savage(X):
    R = risk_matrix(X)
    max_risk = R.max(axis=1)
    return R, max_risk, best_indices(max_risk, maximize=False)


def hurwicz(X, alpha):
    values = alpha * X.min(axis=1) + (1 - alpha) * X.max(axis=1)
    values = np.round(values, 10)  # без этого при a=0.9 у A1 получалось -8.9e-16 вместо 0
    return values, best_indices(values, maximize=True)


def hurwicz_table(X, alphas=ALPHAS):
    # строка = одно значение alpha, столбец = стратегия
    return np.array([hurwicz(X, a)[0] for a in alphas])


def hurwicz_switch_points(X):
    # H_i(a) = max_i - a*(max_i - min_i) - это прямая, поэтому точку смены лидера
    # можно найти точно, приравняв две прямые, а не искать перебором с шагом 0.1
    mins, maxs = X.min(axis=1), X.max(axis=1)
    m = X.shape[0]
    points = []
    for i in range(m):
        for k in range(i + 1, m):
            slope_i, slope_k = maxs[i] - mins[i], maxs[k] - mins[k]
            if abs(slope_i - slope_k) < 1e-12:
                continue
            a = (maxs[i] - maxs[k]) / (slope_i - slope_k)
            if 0 <= a <= 1:
                points.append((a, i, k))
    # пересечений много, но интересны только те, где пересекаются две лучшие стратегии
    # (например A2 и A4 пересекаются при a=0.86, но там обе хуже A3)
    result = []
    for a, i, k in sorted(points):
        vals = hurwicz(X, a)[0]
        if abs(vals[i] - vals.max()) < 1e-9 and abs(vals[k] - vals.max()) < 1e-9:
            result.append((float(a), i, k))
    return result


def drift_robustness(X, noise_levels, n_trials=2000, alpha=0.5, seed=42):
    # Проверка, насколько выбор держится, если оценки в матрице неточные (дрейф данных).
    # Каждый раз шумлю матрицу и смотрю, кто победил по каждому критерию.
    # Шум считаю от размаха матрицы, а не в процентах от элемента: у меня есть 0 и -5,
    # и процент от них почти ничего не меняет.
    # Если победителей двое, каждому идёт по половине.
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    criteria = {
        "Вальд": lambda Y: wald(Y)[1],
        "Максимакс": lambda Y: maximax(Y)[1],
        "Сэвидж": lambda Y: savage(Y)[2],
        f"Гурвиц (α={alpha})": lambda Y: hurwicz(Y, alpha)[1],
    }
    result = {name: np.zeros((len(noise_levels), m)) for name in criteria}
    span = X.max() - X.min()
    for li, level in enumerate(noise_levels):
        for _ in range(n_trials):
            Y = X + rng.uniform(-level, level, size=X.shape) * span
            for name, fn in criteria.items():
                winners = fn(Y)
                for w in winners:
                    result[name][li, w] += 1.0 / len(winners)
        for name in criteria:
            result[name][li] /= n_trials
    return result


def full_report(X, alpha=0.5):
    # всё сразу одним словарём, чтобы в окне не вызывать каждую функцию отдельно
    w_vals, w_best = wald(X)
    mm_vals, mm_best = maximax(X)
    R, s_vals, s_best = savage(X)
    h_vals, h_best = hurwicz(X, alpha)
    return {
        "wald": (w_vals, w_best),
        "maximax": (mm_vals, mm_best),
        "risk": R,
        "savage": (s_vals, s_best),
        "hurwicz": (h_vals, h_best),
        "hurwicz_table": hurwicz_table(X),
        "switch": hurwicz_switch_points(X),
    }


def names(indices, prefix="A"):
    return ", ".join(f"{prefix}{i + 1}" for i in indices)


# графики

def plot_criteria_bars(fig, X, alpha=0.5):
    # 4 столбчатых графика, лучшая стратегия красным
    rep = full_report(X, alpha)
    m = X.shape[0]
    labels = [f"A{i + 1}" for i in range(m)]
    panels = [
        ("Вальд (min строки, берём наибольший)", rep["wald"][0], rep["wald"][1]),
        ("Максимакс (max строки, берём наибольший)", rep["maximax"][0], rep["maximax"][1]),
        ("Сэвидж (max риска, берём наименьший)", rep["savage"][0], rep["savage"][1]),
        (f"Гурвиц при α={alpha}", rep["hurwicz"][0], rep["hurwicz"][1]),
    ]
    fig.clear()
    for k, (title, vals, best) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, k)
        colors = ["#d9534f" if i in best else "#5b8dd6" for i in range(m)]
        bars = ax.bar(labels, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()


def plot_hurwicz(fig, X):
    table = hurwicz_table(X)
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.plot(ALPHAS, table[:, i], marker="o", label=f"A{i + 1}")
    for a, i, k in hurwicz_switch_points(X):
        ax.axvline(a, color="gray", ls="--", lw=0.8)
        ax.annotate(f"α*={a:.2f}\nA{i + 1}=A{k + 1}", (a, ax.get_ylim()[1]),
                    ha="center", va="top", fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("α  (1 = Вальд, пессимист;  0 = максимакс, оптимист)")
    ax.set_ylabel("H(α)")
    ax.set_xticks(ALPHAS)
    ax.set_title("Чувствительность альтернатив по критерию Гурвица")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()


def plot_drift(fig, X, noise_levels=None, n_trials=2000, alpha=0.5):
    # шум от 0 до 30 % с шагом 5
    if noise_levels is None:
        noise_levels = np.round(np.arange(0.0, 0.31, 0.05), 2)
    data = drift_robustness(X, noise_levels, n_trials=n_trials, alpha=alpha)
    fig.clear()
    for k, (name, shares) in enumerate(data.items(), start=1):
        ax = fig.add_subplot(2, 2, k)
        for i in range(X.shape[0]):
            ax.plot(noise_levels * 100, shares[:, i] * 100, marker="o",
                    ms=3, label=f"A{i + 1}")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(-3, 103)
        ax.set_xlabel("уровень дрейфа, % размаха", fontsize=8)
        ax.set_ylabel("доля побед, %", fontsize=8)
        ax.grid(alpha=0.3)
        if k == 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return data


# окно

def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("ЛР 6. Принятие решений в условиях неопределённости")
    root.geometry("1150x780")

    left = ttk.Frame(root, padding=8)
    left.pack(side=tk.LEFT, fill=tk.Y)
    right = ttk.Notebook(root)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # размер матрицы
    size_frame = ttk.Frame(left)
    size_frame.pack(anchor="w")
    ttk.Label(size_frame, text="Стратегий (m):").grid(row=0, column=0)
    m_var = tk.IntVar(value=DEFAULT_MATRIX.shape[0])
    ttk.Spinbox(size_frame, from_=2, to=8, width=4, textvariable=m_var).grid(row=0, column=1)
    ttk.Label(size_frame, text="Состояний (n):").grid(row=0, column=2, padx=(8, 0))
    n_var = tk.IntVar(value=DEFAULT_MATRIX.shape[1])
    ttk.Spinbox(size_frame, from_=2, to=8, width=4, textvariable=n_var).grid(row=0, column=3)

    grid_frame = ttk.LabelFrame(left, text="Матрица полезности X", padding=6)
    grid_frame.pack(anchor="w", pady=6)
    entries = []

    def build_grid(matrix=None):
        for w in grid_frame.winfo_children():
            w.destroy()
        entries.clear()
        m, n = m_var.get(), n_var.get()
        for j in range(n):
            ttk.Label(grid_frame, text=f"S{j + 1}").grid(row=0, column=j + 1)
        for i in range(m):
            ttk.Label(grid_frame, text=f"A{i + 1}").grid(row=i + 1, column=0)
            row = []
            for j in range(n):
                e = ttk.Entry(grid_frame, width=6, justify="center")
                value = matrix[i, j] if matrix is not None and i < matrix.shape[0] \
                    and j < matrix.shape[1] else 0
                e.insert(0, f"{value:g}")
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                row.append(e)
            entries.append(row)

    def read_matrix():
        data = []
        for i, row in enumerate(entries):
            values = []
            for j, e in enumerate(row):
                text = e.get().strip().replace(",", ".")
                try:
                    values.append(float(text))
                except ValueError:
                    raise ValueError(f"Ячейка A{i + 1}/S{j + 1}: '{text}' не число")
            data.append(values)
        return np.array(data, dtype=float)

    btns = ttk.Frame(left)
    btns.pack(anchor="w")
    ttk.Button(btns, text="Изменить размер",
               command=lambda: build_grid(safe_matrix())).pack(side=tk.LEFT)
    ttk.Button(btns, text="Вариант 1",
               command=lambda: (m_var.set(4), n_var.set(4),
                                build_grid(DEFAULT_MATRIX))).pack(side=tk.LEFT, padx=4)

    alpha_frame = ttk.Frame(left)
    alpha_frame.pack(anchor="w", pady=6)
    ttk.Label(alpha_frame, text="α для Гурвица:").pack(side=tk.LEFT)
    alpha_var = tk.DoubleVar(value=0.5)
    alpha_label = ttk.Label(alpha_frame, text="0.5", width=4)
    ttk.Scale(alpha_frame, from_=0, to=1, variable=alpha_var, length=140,
              command=lambda v: alpha_label.config(
                  text=f"{round(float(v), 1):.1f}")).pack(side=tk.LEFT)
    alpha_label.pack(side=tk.LEFT)

    text = tk.Text(left, width=52, height=26, font=("Consolas", 9))
    text.pack(fill=tk.BOTH, expand=True)

    # справа три вкладки с графиками
    tabs = {}
    for key, title in [("bars", "Критерии"), ("hurwicz", "Гурвиц(α)"),
                       ("drift", "Устойчивость к дрейфу")]:
        frame = ttk.Frame(right)
        right.add(frame, text=title)
        fig = Figure(figsize=(7, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tabs[key] = (fig, canvas)

    def safe_matrix():
        try:
            return read_matrix()
        except ValueError:
            return None

    def calculate():
        try:
            X = read_matrix()
        except ValueError as err:
            messagebox.showerror("Ошибка ввода", str(err))
            return
        alpha = round(alpha_var.get(), 1)
        rep = full_report(X, alpha)
        m, n = X.shape
        out = []
        out.append("Исходная матрица X:")
        out.append(format_matrix(X))
        out.append(f"\n1) Вальд: min по строкам = {fmt(rep['wald'][0])}")
        out.append(f"   Оптимум: {names(rep['wald'][1])} (W = {rep['wald'][0].max():g})")
        out.append(f"\n2) Максимакс: max по строкам = {fmt(rep['maximax'][0])}")
        out.append(f"   Оптимум: {names(rep['maximax'][1])} "
                   f"(M = {rep['maximax'][0].max():g})")
        out.append(f"\n3) Сэвидж. max по столбцам = {fmt(X.max(axis=0))}")
        out.append("   Матрица рисков R:")
        out.append(format_matrix(rep["risk"]))
        out.append(f"   max риска по строкам = {fmt(rep['savage'][0])}")
        out.append(f"   Оптимум: {names(rep['savage'][1])} "
                   f"(S = {rep['savage'][0].min():g})")
        out.append(f"\n4) Гурвиц, α={alpha}: H = {fmt(rep['hurwicz'][0])}")
        out.append(f"   Оптимум: {names(rep['hurwicz'][1])}")
        out.append("\nГурвиц при α = 0.0 … 1.0 (шаг 0.1):")
        header = "  α   " + "".join(f"{'A' + str(i + 1):>8}" for i in range(m)) + "  лучший"
        out.append(header)
        for a, row in zip(ALPHAS, rep["hurwicz_table"]):
            out.append(f" {a:.1f} " + "".join(f"{v:8.1f}" for v in row)
                       + f"  {names(best_indices(row))}")
        if rep["switch"]:
            out.append("\nТочки смены лидера по Гурвицу:")
            for a, i, k in rep["switch"]:
                out.append(f"  α* = {a:.3f}: A{i + 1} = A{k + 1}")
        text.delete("1.0", tk.END)
        text.insert(tk.END, "\n".join(out))

        plot_criteria_bars(tabs["bars"][0], X, alpha)
        plot_hurwicz(tabs["hurwicz"][0], X)
        plot_drift(tabs["drift"][0], X, n_trials=600, alpha=alpha)  # 600, а не 2000, иначе окно думает пару секунд
        for fig, canvas in tabs.values():
            canvas.draw()

    ttk.Button(left, text="Рассчитать", command=calculate).pack(fill=tk.X, pady=4)

    build_grid(DEFAULT_MATRIX)
    calculate()
    root.mainloop()


def fmt(values):
    return "[" + ", ".join(f"{v:g}" for v in values) + "]"


def format_matrix(M):
    m, n = M.shape
    lines = ["      " + "".join(f"{'S' + str(j + 1):>7}" for j in range(n))]
    for i in range(m):
        lines.append(f"  A{i + 1}  " + "".join(f"{v:7g}" for v in M[i]))
    return "\n".join(lines)


if __name__ == "__main__":
    run_gui()
