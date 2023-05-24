from itertools import zip_longest

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_trace(fig, x, series, add_trace_kwargs, scatter_kwargs, secondary_y=False):
    if series.name in ["real_price", "real_amount"]:
        mode = "markers"
    else:
        mode = "lines"

    fig.add_trace(go.Scatter(x=x, y=series, name=series.name, mode=mode, **scatter_kwargs), **add_trace_kwargs)

    if secondary_y:
        if series.name == "zscore":
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y2",
                x0=0,
                y0=1,
                x1=1,
                y1=-1,
                fillcolor="gray",
                opacity=0.2,
                layer="below",
                line_width=0,
            )


def update_layout(fig, update_layout_kwargs, secondary_y=False):
    fig.update_layout(
        height=800,
        width=1800,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )

    fig.update_layout(**update_layout_kwargs)

    if secondary_y:
        fig.update_layout(yaxis2=(dict(overlaying="y", side="right")))


def standardize_x(df, x):
    if isinstance(x, str):
        x = df[x]
    elif isinstance(x, pd.Series) or isinstance(x, pd.Index):
        pass
    else:
        raise ValueError("Unknown type for x")
    return x


def standardize_y(df, y):
    if isinstance(y, str):
        y = [df[y]]
    elif isinstance(y, list) or isinstance(y, pd.DataFrame):
        y = [df[col] for col in y]
    elif isinstance(y, pd.Series):
        y = [y]
    else:
        raise ValueError("Unknown type for y")
    return y


def create_fig(df, x, y, multi_y=None, add_trace_kwargs=[{}], scatter_kwargs=[{}], update_layout_kwargs={}):
    if isinstance(add_trace_kwargs, dict):
        add_trace_kwargs = [add_trace_kwargs]
    elif not isinstance(add_trace_kwargs, list):
        raise ValueError("Unknown type for add_trace_kwargs")

    if isinstance(scatter_kwargs, dict):
        scatter_kwargs = [scatter_kwargs]
    elif not isinstance(scatter_kwargs, list):
        raise ValueError("Unknown type for scatter_kwargs")

    if multi_y is None:
        multi_y = []

    fig = go.Figure()

    x = standardize_x(df, x)
    y = standardize_y(df, y)

    for series, add_trace_kwargs, scatter_kwargs in zip_longest(y, scatter_kwargs, add_trace_kwargs, fillvalue={}):
        if series.name in multi_y:
            add_trace(fig, x, series, add_trace_kwargs, {**scatter_kwargs, "yaxis": "y2"}, secondary_y=True)
        else:
            add_trace(fig, x, series, add_trace_kwargs, scatter_kwargs)

    update_layout(fig, update_layout_kwargs, secondary_y=len(multi_y) > 0)

    return fig


def create_subplots_fig(
    df, x, y, row_heights=None, multi_y=None, add_trace_kwargs=[[{}]], scatter_kwargs=[[{}]], update_layout_kwargs={}
):
    assert isinstance(y, list), "Each element of y must be a list of columns to plot on the same subplot"
    if isinstance(add_trace_kwargs, dict):
        add_trace_kwargs = [[add_trace_kwargs]]
    elif isinstance(add_trace_kwargs, list) and isinstance(add_trace_kwargs[0], dict):
        add_trace_kwargs = [add_trace_kwargs]
    elif not isinstance(add_trace_kwargs, list) and not isinstance(add_trace_kwargs[0], list):
        raise ValueError("Unknown type for add_trace_kwargs")

    if isinstance(scatter_kwargs, dict):
        scatter_kwargs = [[scatter_kwargs]]
    elif isinstance(scatter_kwargs, list) and isinstance(scatter_kwargs[0], dict):
        scatter_kwargs = [scatter_kwargs]
    elif not isinstance(scatter_kwargs, list) and not isinstance(scatter_kwargs[0], list):
        raise ValueError("Unknown type for scatter_kwargs")

    if multi_y is None:
        multi_y = [[]] * len(y)
    else:
        assert len(multi_y) == len(y), "multi_y must be the same length as y"

    n_subplots = len(y)
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        specs=[[{"secondary_y": len(multi_y_i) > 0}] for multi_y_i in multi_y],
    )
    x = standardize_x(df, x)
    y = [standardize_y(df, y_i) for y_i in y]
    for (i, subplot), subplot_trace_kwargs, subplot_scatter_kwargs in zip_longest(
        enumerate(y), add_trace_kwargs, scatter_kwargs, fillvalue=[{}]
    ):
        for (
            series,
            add_trace_kwargs,
            scatter_kwargs,
        ) in zip_longest(subplot, subplot_trace_kwargs, subplot_scatter_kwargs, fillvalue={}):
            add_trace(
                fig,
                x,
                series,
                {**add_trace_kwargs, "row": i + 1, "col": 1, "secondary_y": series.name in multi_y[i]},
                scatter_kwargs,
                secondary_y=len(multi_y[i]) > 0,
            )

    update_layout(fig, update_layout_kwargs)

    return fig


def plot_df(df, x, y, multi_y=None, add_trace_kwargs=[{}], scatter_kwargs=[{}], update_layout_kwargs={}):
    fig = create_fig(df, x, y, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    fig.show()


def plot_series(series, add_trace_kwargs=[{}], scatter_kwargs=[{}], update_layout_kwargs={}):
    fig = create_fig(series, series.index, series, None, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    fig.show()


def plot_df_subplots(
    df, x, y, row_heights=None, multi_y=None, add_trace_kwargs=[[{}]], scatter_kwargs=[[{}]], update_layout_kwargs={}
):
    fig = create_subplots_fig(df, x, y, row_heights, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    fig.show()
