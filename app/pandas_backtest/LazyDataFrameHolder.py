import pandas as pd

from .. import plotting


class LazyDataFrameHolder:
    def __init__(self, filepath, name=""):
        self.filepath = filepath
        self.name = name

    def __repr__(self):
        return f"ldfh({self.filepath})"

    @staticmethod
    def _get_between(df, start=None, end=None, padding=0):
        start = start or pd.Timestamp("2000-01-01")
        end = end or pd.Timestamp("2099-12-31")
        freq = df.timestamp.diff().median()
        return df[df.timestamp.between(start - freq * padding, end + freq * padding)]

    def get(self, start=None, end=None, padding=0):
        return self._get_between(pd.read_parquet(self.filepath), start, end, padding)

    def plot(self, start=None, end=None, padding=0, df=None):
        df = df or self.get(start, end, padding)
        plotting.plot_df_subplots(
            df=df,
            x="timestamp",
            y=[
                ["price_signal"],
                ["bid_amount", "ask_amount"],
                ["filled"],
            ],
            row_heights=[0.5, 0.2, 0.2],
            update_layout_kwargs={"title": self.name},
        )
