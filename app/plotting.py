import logging
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def add_trace(
    fig: go.Figure,
    x: Union[pd.Series, pd.Index, List[Any]],
    series: pd.Series,
    add_trace_kwargs: Dict[str, Any],
    scatter_kwargs: Dict[str, Any],
) -> None:
    """
    Add a scatter trace to the figure.

    Args:
        fig: The figure to add the trace to.
        x: The x-values for the trace.
        series: The y-values for the trace.
        add_trace_kwargs: Additional keyword arguments for adding the trace.
        scatter_kwargs: Additional keyword arguments for styling the scatter trace.
    """
    fig.add_trace(go.Scatter(x=x, y=series, name=series.name, **scatter_kwargs), **add_trace_kwargs)


def update_layout(fig: go.Figure, update_layout_kwargs: Dict[str, Any], secondary_y: bool = False) -> None:
    """
    Update the layout of the figure.

    Args:
        fig: The figure to update the layout of.
        update_layout_kwargs: Keyword arguments for updating the layout.
        secondary_y: Flag indicating whether to include a secondary y-axis (default: False).
    """
    fig.update_layout(
        height=800,
        width=1800,
        hovermode="x unified",
        hoverlabel={"namelength": -1},
    )

    fig.update_layout(**update_layout_kwargs)

    if secondary_y:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))


def standardize_x(df: pd.DataFrame, x: Union[str, pd.Series, pd.Index]) -> Union[pd.Series, pd.Index]:
    """
    Standardize the x-values.

    Args:
        df: The DataFrame containing the data.
        x: The x-values.

    Returns:
        The standardized x-values.
    """
    if isinstance(x, str):
        x = df[x]
    elif isinstance(x, pd.Series) or isinstance(x, pd.Index):
        pass
    else:
        raise ValueError("Unknown type for x")
    return x


def standardize_y(df: pd.DataFrame, y: Union[str, List[str], pd.Series, pd.DataFrame]) -> List[pd.Series]:
    """
    Standardize the y-values.

    Args:
        df: The DataFrame containing the data.
        y: The y-values.

    Returns:
        The standardized y-values.
    """
    if isinstance(y, str):
        y = [df[y]]
    elif isinstance(y, list) or isinstance(y, pd.DataFrame):
        y = [df[col] for col in y]
    elif isinstance(y, pd.Series):
        y = [y]
    else:
        raise ValueError("Unknown type for y")
    return y


def create_fig(
    df: pd.DataFrame,
    x: Union[str, pd.Series, pd.Index],
    y: Union[str, List[str], pd.Series, pd.DataFrame],
    multi_y: Optional[List[str]] = None,
    add_trace_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    scatter_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    update_layout_kwargs: Dict[str, Any] = {},
) -> go.Figure:
    """
    Create a plotly figure.

    Args:
        df: The DataFrame containing the data.
        x: The x-values.
        y: The y-values.
        multi_y: List of y-values to plot on a secondary y-axis (default: None).
        add_trace_kwargs: Additional keyword arguments for adding traces (default: [{}]).
        scatter_kwargs: Additional keyword arguments for styling scatter traces (default: [{}]).
        update_layout_kwargs: Keyword arguments for updating the layout (default: {}).

    Returns:
        The plotly figure.
    """
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

    for series, add_trace_kwargs, scatter_kwargs in zip_longest(y, add_trace_kwargs, scatter_kwargs, fillvalue={}):
        if series.name in multi_y:
            add_trace(fig, x, series, add_trace_kwargs, {**scatter_kwargs, "yaxis": "y2"})
        else:
            add_trace(fig, x, series, add_trace_kwargs, scatter_kwargs)

    update_layout(fig, update_layout_kwargs, secondary_y=len(multi_y) > 0)

    return fig


def create_subplots_fig(
    df: pd.DataFrame,
    x: Union[str, pd.Series, pd.Index],
    y: List[List[Union[str, pd.Series, pd.DataFrame]]],
    row_heights: Optional[List[float]] = None,
    multi_y: Optional[List[List[str]]] = None,
    add_trace_kwargs: List[List[Dict[str, Any]]] = [[{}]],
    scatter_kwargs: List[List[Dict[str, Any]]] = [[{}]],
    update_layout_kwargs: Dict[str, Any] = {},
) -> go.Figure:
    """
    Create a plotly figure with subplots.

    Args:
        df: The DataFrame containing the data.
        x: The x-values.
        y: The y-values for each subplot.
        row_heights: The heights of each row in the subplots (default: None).
        multi_y: List of lists indicating which y-values to plot on secondary y-axes for each subplot (default: None).
        add_trace_kwargs: Additional keyword arguments for adding traces (default: [[{}]]).
        scatter_kwargs: Additional keyword arguments for styling scatter traces (default: [[{}]]).
        update_layout_kwargs: Keyword arguments for updating the layout (default: {}).

    Returns:
        The plotly figure with subplots.
    """
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
            )

    update_layout(fig, update_layout_kwargs)

    return fig


def plot_df(
    df: pd.DataFrame,
    x: Union[str, pd.Series, pd.Index],
    y: Union[str, List[str], pd.Series, pd.DataFrame],
    multi_y: Optional[List[str]] = None,
    add_trace_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    scatter_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    update_layout_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Plot a DataFrame using plotly.

    Args:
        df: The DataFrame containing the data.
        x: The x-values.
        y: The y-values.
        multi_y: List of y-values to plot on a secondary y-axis (default: None).
        add_trace_kwargs: Additional keyword arguments for adding traces (default: [{}]).
        scatter_kwargs: Additional keyword arguments for styling scatter traces (default: [{}]).
        update_layout_kwargs: Keyword arguments for updating the layout (default: {}).
    """
    fig = create_fig(df, x, y, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    post_plot_hook(fig, y)
    fig.show()


def plot_series(
    series: pd.Series,
    add_trace_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    scatter_kwargs: Union[Dict[str, Any], List[Dict[str, Any]]] = [{}],
    update_layout_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Plot a Series using plotly.

    Args:
        series: The Series containing the data.
        add_trace_kwargs: Additional keyword arguments for adding traces (default: [{}]).
        scatter_kwargs: Additional keyword arguments for styling scatter traces (default: [{}]).
        update_layout_kwargs: Keyword arguments for updating the layout (default: {}).
    """
    fig = create_fig(series, series.index, series, None, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    post_plot_hook(fig, [series])
    fig.show()


def plot_df_subplots(
    df: pd.DataFrame,
    x: Union[str, pd.Series, pd.Index],
    y: List[List[Union[str, pd.Series, pd.DataFrame]]],
    row_heights: Optional[List[float]] = None,
    multi_y: Optional[List[List[str]]] = None,
    add_trace_kwargs: List[List[Dict[str, Any]]] = [[{}]],
    scatter_kwargs: List[List[Dict[str, Any]]] = [[{}]],
    update_layout_kwargs: Dict[str, Any] = {},
) -> None:
    """
    Plot a DataFrame with subplots using plotly.

    Args:
        df: The DataFrame containing the data.
        x: The x-values.
        y: The y-values for each subplot.
        row_heights: The heights of each row in the subplots (default: None).
        multi_y: List of lists indicating which y-values to plot on secondary y-axes for each subplot (default: None).
        add_trace_kwargs: Additional keyword arguments for adding traces (default: [[{}]]).
        scatter_kwargs: Additional keyword arguments for styling scatter traces (default: [[{}]]).
        update_layout_kwargs: Keyword arguments for updating the layout (default: {}).
    """
    fig = create_subplots_fig(df, x, y, row_heights, multi_y, add_trace_kwargs, scatter_kwargs, update_layout_kwargs)
    post_plot_hook(fig, y)
    fig.show()
    

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
