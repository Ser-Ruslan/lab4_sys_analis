"""Запуск: python lab5_variant1.py. Без интерфейса: --cli."""
import argparse
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from lab5_model import DEFAULT_A, DEFAULT_B, DEFAULT_C, solve, sensitivity


def fmt(n):
    return f'{n:,.2f}'.replace(',', ' ').replace('.', ',')


class App:
    def __init__(self, root):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.root = root
        self.current = None
        root.title('Лабораторная работа 5  Производственный план  Вариант 1')
        root.geometry('1120x920')
        root.minsize(1000, 860)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('Treeview', rowheight=27, font=('Segoe UI', 10))
        style.configure('TLabel', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=6)
        top = ttk.Frame(root, padding=(18, 12))
        top.grid(row=0, column=0, sticky='ew')
        ttk.Label(top, text='Оптимизация производственного плана', font=('Segoe UI', 19, 'bold')).pack(anchor='w')
        ttk.Label(top, text='Вариант 1 • max cᵀx при Ax ≤ b, x ≥ 0 • непрерывные объёмы выпуска').pack(anchor='w', pady=(4, 0))
        inputs = ttk.LabelFrame(root, text='Исходные данные  все значения можно изменять', padding=10)
        inputs.grid(row=1, column=0, sticky='ew', padx=18)
        for j, text in enumerate(['Материал / показатель', 'P1', 'P2', 'P3', 'Запас, кг']):
            ttk.Label(inputs, text=text).grid(row=0, column=j, padx=12, pady=3)
        self.variables = []
        check = (root.register(self.allow_value), '%P')
        for i in range(4):
            ttk.Label(inputs, text=f'S{i+1}  расход, кг/ед.' if i < 3 else 'Цена, у. е./ед.').grid(row=i+1, column=0, sticky='w')
            line = []
            for j in range(4 if i < 3 else 3):
                var = tk.StringVar()
                entry = ttk.Entry(inputs, textvariable=var, width=12, validate='key', validatecommand=check)
                entry.grid(row=i+1, column=j+1, padx=12, pady=3, sticky='ew')
                var.trace_add('write', self.invalidate)
                line.append(var)
            self.variables.append(line)
        for j in range(1, 5):
            inputs.columnconfigure(j, weight=1)
        buttons = ttk.Frame(inputs)
        buttons.grid(row=5, column=0, columnspan=5, sticky='w', pady=(9, 0))
        ttk.Button(buttons, text='Рассчитать', command=self.calculate).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='Очистить', command=self.reset).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='Сохранить результат JSON', command=self.export).pack(side='left')
        ttk.Label(buttons, text='«Очистить» восстанавливает вариант 1').pack(side='left', padx=12)
        tabs = ttk.Notebook(root)
        tabs.grid(row=2, column=0, sticky='nsew', padx=18, pady=12)
        result_tab = ttk.Frame(tabs, padding=10)
        sensitivity_tab = ttk.Frame(tabs, padding=10)
        tabs.add(result_tab, text='План и ресурсы')
        tabs.add(sensitivity_tab, text='Анализ чувствительности')
        result_tab.columnconfigure(0, weight=1)
        result_tab.rowconfigure(3, weight=1)
        self.total = ttk.Label(result_tab, text='', font=('Segoe UI', 15, 'bold'))
        self.total.grid(row=0, column=0, sticky='w')
        self.plan = self.table(result_tab, ['Изделие', 'Выпуск, ед.', 'Стоимость, у. е.'], 3)
        self.plan.grid(row=1, column=0, sticky='ew', pady=6)
        self.resources = self.table(result_tab, ['Материал', 'Запас, кг', 'Расход, кг', 'Остаток, кг', 'Ограничение', 'Теневая цена'], 3)
        self.resources.grid(row=2, column=0, sticky='ew', pady=6)
        self.figure = Figure(figsize=(9, 2.4), dpi=100, layout='constrained')
        self.axes = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, master=result_tab)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky='nsew')
        ttk.Label(result_tab, text='Активное ограничение: остаток ≈ 0. Экономический дефицит: теневая цена > 0.', wraplength=900).grid(row=4, column=0, sticky='w')
        sensitivity_tab.columnconfigure(0, weight=1)
        self.scenarios = self.table(sensitivity_tab, ['Материал', 'Изменение, кг', 'Новый запас, кг', 'Стоимость, у. е.', 'Прирост, у. е.'], 9)
        self.scenarios.grid(row=0, column=0, sticky='ew')
        ttk.Label(sensitivity_tab, text='Каждая строка  отдельный повторный расчёт: меняется запас только одного материала.\nТеневая цена показывает локальный прирост стоимости при добавлении 1 кг; она может измениться при больших изменениях запасов.', wraplength=900, justify='left').grid(row=1, column=0, sticky='w', pady=15)
        ttk.Label(sensitivity_tab, text='Для исходного варианта Fmax = 1020 у. е. Оптимум не единственный:\n(x1, x2, x3) = (260 − 5t/3, 80 + t/3, t), 60 ≤ t ≤ 156.\nS3 может расходоваться полностью при нулевой теневой цене.\nЭта справка относится только к исходным данным варианта 1.', wraplength=900, justify='left').grid(row=2, column=0, sticky='w')
        self.status = tk.StringVar(value='Готово к расчёту')
        ttk.Label(root, textvariable=self.status, padding=(18, 0, 18, 10)).grid(row=3, column=0, sticky='ew')
        self.reset()

    @staticmethod
    def table(parent, columns, height):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=height)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=130, minwidth=90, anchor='center')
        return tree

    @staticmethod
    def allow_value(text):
        if not text:
            return True
        # Допускаем запятую и промежуточное состояние «.»; минус и буквы блокируются.
        if any(char not in '0123456789.,' for char in text):
            return False
        return text.count('.') + text.count(',') <= 1

    def invalidate(self, *_):
        self.current = None
        if hasattr(self, 'status'):
            self.status.set('Данные изменены  нажмите «Рассчитать»')
            for tree in (self.plan, self.resources, self.scenarios):
                tree.delete(*tree.get_children())
            self.total.configure(text='Результат требует пересчёта')
            self.axes.clear()
            self.canvas.draw_idle()

    def reset(self):
        for i, row in enumerate(self.variables):
            vals = DEFAULT_A[i] + [DEFAULT_B[i]] if i < 3 else DEFAULT_C
            for var, value in zip(row, vals):
                var.set(str(value))
        self.calculate()

    def read_data(self):
        values = []
        for i, row in enumerate(self.variables):
            line = []
            for j, var in enumerate(row):
                try:
                    value = float(var.get().replace(',', '.'))
                    if not math.isfinite(value) or value < 0:
                        raise ValueError()
                except ValueError:
                    raise ValueError(f'Некорректное поле: строка {i+1}, столбец {j+1}. Введите неотрицательное число.')
                line.append(value)
            values.append(line)
        return [r[:3] for r in values[:3]], [r[3] for r in values[:3]], values[3]

    def calculate(self):
        self.invalidate()
        try:
            A, b, c = self.read_data()
            result = solve(A, b, c)
            self.status.set(f'{result.status}  {result.message}')
            if result.status != 'OPTIMAL':
                self.total.configure(text=result.message)
                messagebox.showwarning('Статус оптимизации', result.message, parent=self.root)
                return
            scenarios = sensitivity(A, b, c)
            self.current = {'A': A, 'b': b, 'c': c, 'x': result.x.tolist(),
                            'value': result.value, 'used': result.used.tolist(),
                            'slack': result.slack.tolist(), 'shadow': result.shadow.tolist(),
                            'scenarios': scenarios}
            self.total.configure(text=f'Максимальная стоимость выпуска: {fmt(result.value)} у. е.')
            for i in range(3):
                self.plan.insert('', 'end', values=(f'P{i+1}', fmt(result.x[i]), fmt(c[i]*result.x[i])))
                self.resources.insert('', 'end', values=(f'S{i+1}', fmt(b[i]), fmt(result.used[i]), fmt(result.slack[i]), 'Активно' if result.binding[i] else 'Есть остаток', fmt(result.shadow[i])))
            for row in scenarios:
                self.scenarios.insert('', 'end', values=(f'S{row["resource"]}', fmt(row['delta']), fmt(row['stock']), fmt(row['value']) if row['value'] is not None else row['status'], fmt(row['gain']) if row['gain'] is not None else ' '))
            self.axes.bar(['S1', 'S2', 'S3'], result.used, label='Израсходовано', color='#3779ab')
            self.axes.bar(['S1', 'S2', 'S3'], result.slack, bottom=result.used, label='Остаток', color='#a8cfa1')
            self.axes.set_ylabel('Материал, кг')
            self.axes.set_title('Использование ресурсов')
            self.axes.legend(loc='upper left', ncol=2)
            self.axes.set_ylim(0, max(max(b), 1)*1.25)
            self.canvas.draw_idle()
        except ValueError as exc:
            self.status.set('Ошибка ввода')
            messagebox.showwarning('Проверьте данные', str(exc), parent=self.root)

    def export(self):
        if self.current is None:
            messagebox.showinfo('Нет результата', 'Сначала выполните успешный расчёт.', parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', initialfile='lab5_results.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                Path(path).write_text(json.dumps(self.current, ensure_ascii=False, indent=2), encoding='utf-8')
                self.status.set('Результат сохранён')
            except OSError as exc:
                messagebox.showerror('Ошибка сохранения', str(exc), parent=self.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', action='store_true')
    args = parser.parse_args()
    if args.cli:
        result = solve()
        print(result.status)
        for i, x in enumerate(result.x, 1):
            print(f'P{i} = {x:.2f}')
        print(f'Fmax = {result.value:.2f}')
        for i in range(3):
            print(f'S{i+1}: used={result.used[i]:.2f}; slack={result.slack[i]:.2f}; shadow={result.shadow[i]:.2f}')
    else:
        root = tk.Tk()
        App(root)
        root.mainloop()


if __name__ == '__main__':
    main()
