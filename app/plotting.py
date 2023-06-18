import pyutil


def post_plot_hook(fig, y):
    if "zscore" in y:
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

    marker_scatters = [i for i, _ in enumerate(y) if y.name in ["real_price", "real_amount"]]
    for i in marker_scatters:
        fig.data[i].mode = "markers"


def plot_df(df, x, y, multi_y=None, add_trace_kwargs=[{}], scatter_kwargs=[{}], update_layout_kwargs={}):
    fig = pyutil.create_fig(df, x, y, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    post_plot_hook(fig, y)
    fig.show()


def plot_series(series, add_trace_kwargs=[{}], scatter_kwargs=[{}], update_layout_kwargs={}):
    fig = pyutil.create_fig(series, series.index, series, None, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    post_plot_hook(fig, [series])
    fig.show()


def plot_df_subplots(
    df, x, y, row_heights=None, multi_y=None, add_trace_kwargs=[[{}]], scatter_kwargs=[[{}]], update_layout_kwargs={}
):
    fig = pyutil.create_subplots_fig(
        df, x, y, row_heights, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs
    )
    post_plot_hook(fig, y)
    fig.show()
