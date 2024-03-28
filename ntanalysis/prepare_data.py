import math
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm


COLUMN_TIME = "TIME"
EPOCH_TIME = datetime(1970, 1, 1)

COLUMN_TRANSLATOR = {
    "Дата Время": COLUMN_TIME,
}


class SUFFIX:
    TIME_SECONDS = "_seconds"
    TIME_DAYOFWEEK = "_dow"


class Preprocessor:
    def __init__(self):
        self.data = None

    @staticmethod
    def from_csv(filename):
        preprocessor = Preprocessor()
        preprocessor.data = pd.read_csv(
            filename, skiprows=2, encoding="cp1251", on_bad_lines="skip"
        )
        return preprocessor

    def rename_columns(self):
        self.data = self.data.rename(columns=COLUMN_TRANSLATOR)
        self.data = self.data.set_index(COLUMN_TIME)
        return self

    def to_datatypes(self):
        # time column - to datetime
        self.data.index = pd.to_datetime(pd.to_numeric(self.data.index) // 1000, unit="s")
        # other columns to float, ignoring weird values
        for col in tqdm(self.data.columns):
            self.data[col] = self.data[col].apply(pd.to_numeric, errors="coerce")
        return self

    def normalize(self):
        for col in self.data.select_dtypes(include=[np.float64]):
            self.data[col] = (self.data[col] - self.data[col].mean()) / math.sqrt(
                self.data[col].var()
            )

    def fill_nans(self):
        self.data.interpolate(inplace=True, method="time")
        self.data.fillna(inplace=True, method="bfill")
        self.data = self.data.dropna(axis=1, how="all")

    def sort_columns(self):
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)

    def sort_index(self):
        self.data.sort_index(inplace=True)

    def expand_columns(self):
        self.data[COLUMN_TIME + SUFFIX.TIME_SECONDS] = self.data.apply(
            lambda row: (row.name - EPOCH_TIME).total_seconds(), axis=1
        )
        self.data[COLUMN_TIME + SUFFIX.TIME_DAYOFWEEK] = self.data.apply(
            lambda row: row.name.weekday(), axis=1
        )

    def to_csv(self, csv_path):
        self.data.to_csv(csv_path)

    def plot(self):
        self.data.plot(y=[col for col in self.data if not col.startswith(COLUMN_TIME)])
        plt.show()


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def prepare(cfg: DictConfig):
    # read
    preprocessor = Preprocessor.from_csv(cfg.data.raw_csv)
    preprocessor.rename_columns()
    print(preprocessor.data)
    # process data
    preprocessor.to_datatypes()
    preprocessor.normalize()
    preprocessor.sort_columns()
    preprocessor.sort_index()
    preprocessor.fill_nans()
    preprocessor.expand_columns()
    print(preprocessor.data)
    # save to file
    preprocessor.to_csv(cfg.data.csv_path)
    # plot
    preprocessor.plot()


if __name__ == "__main__":
    prepare()
