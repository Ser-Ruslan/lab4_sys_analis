"""Лабораторная № 4. Запуск: python lab4_variant1.py. Только стандартная библиотека."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from lab4_model import Parameters, VARIANTS, calculate, costs, inventory_points, schedule


def number(value):
    return f'{value:,.2f}'.replace(',', ' ').replace('.', ',')


def explanation(p, r):
    return (
        f'В среднем требуется {number(r.daily_demand)} единицы товара в день. '
        f'Выгодный размер партии: {number(r.Q)} ед., для целого товара - {r.integer_Q} ед. '
        f'За год получается {number(r.N)} заказа, между заказами проходит {number(r.T)} дня.\n\n'
        f'Новую заявку нужно подавать при позиции запаса {number(r.ROP)} ед. '
        f'За {number(p.L)} дня доставки расходуется ровно столько товара. '
        'Позиция запаса - остаток на складе плюс уже заказанный товар. '
        'Если в пути нет других партий, она равна остатку на складе.\n\n'
        f'Заказы и хранение стоят {number(r.management)} руб. в год, '
        f'закупка - {number(r.purchase)} руб. Общие затраты: {number(r.total)} руб. '
        'При постоянной цене закупка не зависит от размера партии. '
        'Слишком маленькие партии увеличивают расходы на заказы, '
        'а слишком большие - расходы на хранение.\n\n'
        'Модель предполагает постоянный спрос, точные сроки доставки и отсутствие дефицита. '
        'Страховой запас здесь не учитывается. Дробное число заказов - средняя частота '
        'в длительной работе склада, а не число заявок в конкретном календарном году.'
    )


def export(p, r, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'results.json').write_text(
        json.dumps({'parameters': asdict(p), 'results': asdict(r)}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    with (folder / 'inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['День', 'Запас, ед.'])
        for day, stock in inventory_points(p, r):
            writer.writerow([f'{day:.8f}', f'{stock:.8f}'])
    with (folder / 'schedule.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, ['cycle', 'order_day', 'arrival_day', 'quantity'], delimiter=';')
        writer.writeheader()
        writer.writerows(schedule(p, r))


class App:
    def __init__(self, root, parameters):
        self.root = root
        root.title('Лабораторная № 4 - Управление запасами')
        root.geometry('1100x780')
        root.minsize(850, 650)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        top = ttk.LabelFrame(root, text='Исходные данные', padding=10)
        top.grid(row=0, column=0, sticky='ew', padx=12, pady=10)
        ttk.Label(top, text='Вариант').grid(row=0, column=0)
        self.variant = tk.StringVar(value='1')
        chooser = ttk.Combobox(top, textvariable=self.variant, values=list(VARIANTS), width=5, state='readonly')
        chooser.grid(row=1, column=0, padx=5)
        chooser.bind('<<ComboboxSelected>>', self.load_variant)
        self.fields = {}
        labels = [('D', 'Спрос, ед./год'), ('Co', 'Заказ, руб.'), ('Ch', 'Хранение, руб./ед./год'),
                  ('L', 'Доставка, дней'), ('Cp', 'Цена, руб./ед.')]
        for col, (key, title) in enumerate(labels, 1):
            ttk.Label(top, text=title).grid(row=0, column=col, padx=6)
            value = tk.StringVar(value=str(getattr(parameters, key)))
            ttk.Entry(top, textvariable=value, width=13).grid(row=1, column=col, padx=6)
            self.fields[key] = value
        ttk.Button(top, text='Рассчитать', command=self.recalculate).grid(row=1, column=6, padx=6)
        ttk.Button(top, text='Сохранить данные', command=self.save).grid(row=1, column=7, padx=6)
        tabs = ttk.Notebook(root)
        tabs.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 12))
        results = ttk.Frame(tabs, padding=12)
        graphs = ttk.Frame(tabs)
        analysis = ttk.Frame(tabs, padding=12)
        tabs.add(results, text='Расчёты и поставки')
        tabs.add(graphs, text='Графики')
        tabs.add(analysis, text='Объяснение')
        self.summary = tk.StringVar()
        ttk.Label(results, textvariable=self.summary, font=('Segoe UI', 11), justify='left').pack(anchor='w')
        ttk.Label(results, text='Три цикла поставки (начальный запас уже на складе)', padding=(0, 18)).pack(anchor='w')
        self.table = ttk.Treeview(results, columns=('cycle', 'order', 'arrival', 'q'), show='headings', height=5)
        for key, title in zip(self.table['columns'], ['Цикл', 'День заказа', 'День поставки', 'Партия, ед.']):
            self.table.heading(key, text=title)
            self.table.column(key, width=160, anchor='center')
        self.table.pack(fill='x')
        ttk.Label(results, text='Время отсчитывается в днях от начала наблюдения; интервалы не округляются.', padding=(0, 12)).pack(anchor='w')
        self.canvas = tk.Canvas(graphs, background='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.text = tk.Text(analysis, wrap='word', font=('Segoe UI', 12), padx=10, pady=10)
        self.text.pack(fill='both', expand=True)
        self.p = parameters
        self.r = calculate(parameters)
        self.recalculate()

    def load_variant(self, event=None):
        for key, value in zip(self.fields, VARIANTS[int(self.variant.get())]):
            self.fields[key].set(str(value))
        self.recalculate()

    def recalculate(self):
        try:
            p = Parameters(**{key: float(value.get().replace(',', '.')) for key, value in self.fields.items()})
            r = calculate(p)
        except (ValueError, OverflowError) as error:
            messagebox.showerror('Проверьте данные', str(error))
            return False
        self.p, self.r = p, r
        self.summary.set(
            f'Оптимальная партия Q*: {number(r.Q)} ед.   |   Целая партия: {r.integer_Q} ед.\n'
            f'Заказов в год N*: {number(r.N)}   |   Интервал T*: {number(r.T)} дня\n'
            f'Точка заказа ROP: {number(r.ROP)} ед.   |   Спрос в день: {number(r.daily_demand)} ед.\n\n'
            f'Оформление заказов: {number(r.ordering)} руб./год\n'
            f'Хранение: {number(r.holding)} руб./год\n'
            f'Закупка: {number(r.purchase)} руб./год\n'
            f'Заказы и хранение: {number(r.management)} руб./год\n'
            f'Общие годовые затраты TC: {number(r.total)} руб.')
        self.table.delete(*self.table.get_children())
        for row in schedule(p, r):
            self.table.insert('', 'end', values=(row['cycle'], number(row['order_day']), number(row['arrival_day']), number(row['quantity'])))
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', explanation(p, r))
        self.text.configure(state='disabled')
        self.draw()
        return True

    def draw(self):
        if not hasattr(self, 'r'):
            return
        c, p, r = self.canvas, self.p, self.r
        c.delete('all')
        w, h = max(c.winfo_width(), 800), max(c.winfo_height(), 550)
        left, right = 90, w - 30
        top, bottom = 65, h / 2 - 18
        position = p.L >= r.T
        ceiling = r.Q + r.ROP if position else r.Q * 1.13
        x = lambda day: left + day / (3 * r.T) * (right - left)
        y = lambda stock: bottom - stock / ceiling * (bottom - top)
        title = 'Позиция запаса при долгой доставке' if position else 'Запас на складе и момент заказа'
        c.create_text(w / 2, 22, text=title, font=('Segoe UI', 13, 'bold'))
        for i in range(5):
            value = ceiling * i / 4
            c.create_line(left, y(value), right, y(value), fill='#dddddd')
            c.create_text(left - 8, y(value), text=number(value), anchor='e')
        c.create_line(left, top, left, bottom, right, bottom)
        c.create_text(left, 43, text='Единицы', anchor='w')
        if position:
            # Позиция растёт в момент заказа; заказы имеют период T.
            phase = (p.L % r.T)
            first = r.T - phase if phase else r.T
            points = [(0, r.ROP + r.daily_demand * first)]
            for k in range(4):
                t = first + k * r.T
                if t <= 3 * r.T:
                    points.extend([(t, r.ROP), (t, r.ROP + r.Q)])
            points.append((3 * r.T, points[-1][1] - r.daily_demand * (3 * r.T - points[-1][0])))
        else:
            points = inventory_points(p, r)
        c.create_line(*[v for t, s in points for v in (x(t), y(s))], fill='#2463a0', width=2)
        c.create_line(left, y(r.ROP), right, y(r.ROP), fill='#b23b32', dash=(6, 4), width=2)
        c.create_text(right, y(r.ROP) - 10, text=f'ROP = {number(r.ROP)}', anchor='e', fill='#b23b32')
        for row in schedule(p, r):
            day = row['order_day']
            if 0 <= day <= 3 * r.T:
                c.create_oval(x(day)-4, y(r.ROP)-4, x(day)+4, y(r.ROP)+4, fill='#b23b32', outline='')
            c.create_text(x(row['arrival_day']), bottom + 15, text=number(row['arrival_day']))
        c.create_text(left, bottom + 15, text='0')
        c.create_text(w / 2, bottom + 36, text='Дни; красные точки - отправка заказа')
        low_top, low_bottom = h / 2 + 75, h - 48
        c.create_text(w / 2, h / 2 + 38, text='Годовые расходы на заказы и хранение', font=('Segoe UI', 13, 'bold'))
        min_q, max_q = r.Q / 4, r.Q * 2
        data = [(min_q + (max_q-min_q)*i/160) for i in range(161)]
        values = [costs(p, q)[0] + costs(p, q)[1] for q in data]
        max_cost = max(values) * 1.08
        cx = lambda q: left + (q-min_q)/(max_q-min_q)*(right-left)
        cy = lambda value: low_bottom - value / max_cost * (low_bottom-low_top)
        for i in range(5):
            val = max_cost * i / 4
            c.create_line(left, cy(val), right, cy(val), fill='#dddddd')
            c.create_text(left-8, cy(val), text=f'{val:.0f}', anchor='e')
        c.create_line(left, low_top, left, low_bottom, right, low_bottom)
        c.create_line(*[v for q, value in zip(data, values) for v in (cx(q), cy(value))], fill='#2463a0', width=2)
        c.create_oval(cx(r.Q)-4, cy(r.management)-4, cx(r.Q)+4, cy(r.management)+4, fill='#b23b32')
        c.create_text(cx(r.Q), cy(r.management)-18, text=f'Минимум: Q = {number(r.Q)}')
        for q in [min_q, r.Q, max_q]:
            c.create_text(cx(q), low_bottom+15, text=number(q))
        c.create_text(w/2, h-10, text='Размер партии, ед. (закупка одинакова для всех партий)')

    def save(self):
        if not self.recalculate():
            return
        folder = filedialog.askdirectory(title='Папка для результатов')
        if folder:
            export(self.p, self.r, folder)
            messagebox.showinfo('Готово', 'Сохранены расчёты JSON, график в CSV и расписание поставок в CSV.')


def main():
    parser = argparse.ArgumentParser(description='Модель Уилсона - лабораторная № 4')
    parser.add_argument('--variant', type=int, choices=VARIANTS, default=1)
    parser.add_argument('--no-gui', action='store_true', help='Расчёт без окна')
    parser.add_argument('--output', default='output/lab4_variant1')
    for key in ('D', 'Co', 'Ch', 'L', 'Cp'):
        parser.add_argument('--' + key, type=float)
    args = parser.parse_args()
    values = dict(zip(('D', 'Co', 'Ch', 'L', 'Cp'), VARIANTS[args.variant]))
    values.update({k: getattr(args, k) for k in values if getattr(args, k) is not None})
    p = Parameters(**values)
    try:
        r = calculate(p)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.no_gui:
        export(p, r, args.output)
        print(explanation(p, r))
    else:
        root = tk.Tk()
        app = App(root, p)
        app.variant.set(str(args.variant))
        root.mainloop()


if __name__ == '__main__':
    main()
