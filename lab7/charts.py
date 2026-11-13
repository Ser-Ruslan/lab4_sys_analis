# ЛР 7. Интерактивные графики Plotly: пространство альтернатив (2D/3D) и столбики расстояний.
# Функции ничего не считают сами, им передаются уже готовые маски и координаты.

import numpy as np
import plotly.graph_objects as go

import mcda

COLOR_PARETO = "#1f77b4"
COLOR_DOMINATED = "#9aa0a6"
COLOR_IDEAL = "#e6a700"
COLOR_BEST = "#d62728"
COLOR_CONCESSION = "#2ca02c"


def _hover(names, values, criteria):
    # в подсказке показываю все критерии в исходных единицах, даже если на осях только часть или z
    lines = []
    for name, row in zip(names, values):
        parts = [f"<b>{name}</b>"] + [f"{c}: {v:g}" for c, v in zip(criteria, row)]
        lines.append("<br>".join(parts))
    return lines


def alternatives_scatter(names, coords, axis_titles, pareto, ideal, hover, marks=None, front=None):
    """
    coords      - n x 2 или n x 3, от числа столбцов зависит, 2D это или 3D
    pareto      - маска Парето-эффективных альтернатив
    ideal       - координаты идеальной точки в тех же осях
    marks       - {подпись: (список индексов, цвет)} - победители методов, обводятся кольцом
    front       - для 2D: индексы точек границы Парето в этой проекции (соединяются линией)
    """
    coords = np.asarray(coords, dtype=float)
    dims = coords.shape[1]
    is3d = dims == 3
    Scatter = go.Scatter3d if is3d else go.Scatter
    pareto = np.asarray(pareto, dtype=bool)
    hover = np.asarray(hover, dtype=object)
    names = np.asarray(names, dtype=object)

    def xyz(points):
        points = np.atleast_2d(points)
        kw = {"x": points[:, 0], "y": points[:, 1]}
        if is3d:
            kw["z"] = points[:, 2]
        return kw

    fig = go.Figure()

    # граница Парето: в 3D полупрозрачная поверхность по эффективным точкам, в 2D ломаная
    if is3d and pareto.sum() >= 3:
        p = coords[pareto]
        fig.add_trace(go.Mesh3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], color=COLOR_PARETO, opacity=0.15,
                                name="Граница Парето", hoverinfo="skip", showlegend=True))
    if not is3d and front is not None and len(front) >= 2:
        p = coords[sorted(front, key=lambda i: coords[i, 0])]
        fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines", name="Граница Парето в этой проекции",
                                 line=dict(color=COLOR_PARETO, dash="dash", width=2), hoverinfo="skip"))

    for mask, label, color, size in ((~pareto, "Доминируемые", COLOR_DOMINATED, 9),
                                     (pareto, "Парето-эффективные", COLOR_PARETO, 11)):
        if not mask.any():
            continue
        fig.add_trace(Scatter(**xyz(coords[mask]), mode="markers+text", name=label,
                              text=names[mask], textposition="top center",
                              hovertext=hover[mask], hoverinfo="text",
                              marker=dict(size=size if not is3d else size - 3, color=color,
                                          line=dict(color="white", width=1))))

    # победители методов: пустые кольца разного размера, чтобы было видно, если это одна и та же точка
    for k, (label, (idx, color)) in enumerate((marks or {}).items()):
        if not idx:
            continue
        fig.add_trace(Scatter(**xyz(coords[idx]), mode="markers", name=label, hoverinfo="skip",
                              marker=dict(size=(12 + 5 * k) if is3d else (20 + 8 * k), color=color,
                                          symbol="circle-open", line=dict(color=color, width=3))))

    fig.add_trace(Scatter(**xyz(ideal), mode="markers+text", name="Идеальная точка", text=["Идеал"],
                          textposition="top center", hovertext=["<b>Идеальная точка</b>"], hoverinfo="text",
                          marker=dict(size=8 if is3d else 18, color=COLOR_IDEAL,
                                      symbol="diamond" if is3d else "star",
                                      line=dict(color="black", width=1))))

    layout = dict(margin=dict(l=10, r=10, t=30, b=10), height=560 if is3d else 460,
                  legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    if is3d:
        # камера сбоку: при стандартной (1.25, 1.25, 1.25) точки сливаются, потому что в таких задачах
        # критерии обычно растут вместе (точнее = медленнее = дороже) и облако вытянуто как раз вдоль взгляда
        layout["scene"] = dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1], zaxis_title=axis_titles[2],
                               aspectmode="cube", camera=dict(eye=dict(x=1.7, y=-1.5, z=0.8)))
    else:
        layout["xaxis_title"], layout["yaxis_title"] = axis_titles[0], axis_titles[1]
    fig.update_layout(**layout)
    return fig


def projection_front(values, maximize, axes):
    # граница Парето только по двум выбранным критериям (у проекции она своя, не как в полном пространстве)
    mask = mcda.pareto_mask(np.asarray(values)[:, axes], np.asarray(maximize)[axes])
    return [int(i) for i in np.where(mask)[0]]


def distance_bars(names, distances, pareto):
    order = np.argsort(distances)
    colors = [COLOR_BEST if k == 0 else (COLOR_PARETO if pareto[i] else COLOR_DOMINATED)
              for k, i in enumerate(order)]
    fig = go.Figure(go.Bar(x=[names[i] for i in order], y=[distances[i] for i in order], marker_color=colors,
                           text=[f"{distances[i]:.4f}" for i in order], textposition="outside"))
    fig.update_layout(yaxis_title="D(x) — расстояние до идеала", height=380,
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    return fig


def hover_text(names, values, criteria):
    return _hover(names, values, criteria)
