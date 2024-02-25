import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from tqdm import tqdm


FILENAME_IN = "data/raw.xlsx"
FILENAME_OUT = "data/out.csv"
DEBUG_ONLY_FIRST_N_LINES = 999999
if DEBUG_ONLY_FIRST_N_LINES is not None:
    print(f"WARNING only {DEBUG_ONLY_FIRST_N_LINES=} lines will be processed")
COLUMN_TRANSLATOR1 = {
    "YLNG.33.615II00257.PV": "current",  # Сила тока электродвигателя,А
    "YLNG.33.615VI00505.PV": "vibr_hor_1",  # Вибрация подшипника насоса горизонтальная 1
    "YLNG.33.615VI00506.PV": "vibr_hor_2",  # Вибрация подшипника насоса горизонтальная 2
    "YLNG.33.615TI00513.PV": "temp",  # Температура среды на входе насоса
    "YLNG.33.615PZI00256.PV": "pres_in",  # Давление среды на входе насоса
    "YLNG.33.615PI00295.PV": "pres_out",  # Давление среды после насоса
    "YLNG.33.615PDI00254.PV": "diff",  # Перепад на входном стрейнере
    "YLNG.33.615ZI00270.PV": "pos_high",  # Положение регулирующей арматуры на напоре
    "YLNG.33.615ZI00219.PV": "pos_low",  # Положение регулирующей арматуры min-flow
    "YLNG.33.615FI00270.PV": "consum",  # Расход
}


class SUFFIX:
    TIME = "_time"
    VAL = "_val"
    TIME_SECONDS = "_seconds"
    TIME_DAYOFWEEK = "_dow"


COLUMN_TRANSLATOR2 = {"Время": SUFFIX.TIME, "Значение": SUFFIX.VAL}
COLUMN_TIME = "TIME"
EPOCH_TIME = datetime(1970, 1, 1)


class Preprocessor:
    def __init__(self):
        self.data = None

    @staticmethod
    def from_excel():
        preprocessor = Preprocessor()
        lines = Preprocessor.read_excel_gen(FILENAME_IN)
        preprocessor.data = Preprocessor.gen_to_pd(lines, DEBUG_ONLY_FIRST_N_LINES)
        return preprocessor

    @staticmethod
    def read_excel_gen(filename):
        book = openpyxl.load_workbook(filename=filename, read_only=True, data_only=True)
        first_sheet = book.worksheets[0]
        return first_sheet.values

    @staticmethod
    def gen_to_pd(rows_generator, nrows):
        header_row_main = next(rows_generator)
        header_row_time = next(rows_generator)
        header_row = []
        for i in range(len(header_row_main)):
            x, y = header_row_main[i], header_row_time[i]
            if x is None:
                x = header_row_main[i - 1]
            header_row.append(COLUMN_TRANSLATOR1[x] + COLUMN_TRANSLATOR2[y])
        data_rows = []
        for _, row in tqdm(
            zip(range(nrows - 2), rows_generator, strict=False), total=nrows - 2
        ):
            row = [
                x if x != "Bad Input" and x != "Error! Maximum number of rows" else None
                for x in row
            ]
            data_rows.append(row)
        return pd.DataFrame(data_rows, columns=header_row).iloc[:, 2:]

    def to_common_time(self):
        # create separate tables for each feature
        columns = {}
        for col in set(self.data):
            if col is None or not col.endswith(SUFFIX.TIME):
                continue
            col_pref = col[: -len(SUFFIX.TIME)]
            col_time = col_pref + SUFFIX.TIME
            col_val = col_pref + SUFFIX.VAL
            columns[col_pref] = self.data[[col_time, col_val]]
            columns[col_pref] = columns[col_pref].rename(
                columns={col_time: COLUMN_TIME, col_val: col_pref}
            )
            columns[col_pref].dropna(inplace=True, how="all")
        # join dataframes for common timeline
        result = list(columns.values())[0]
        for dataframe in tqdm(list(columns.values())[1:]):
            result = pd.merge(result, dataframe, on=COLUMN_TIME, how="outer")
        self.data = result.set_index(COLUMN_TIME)
        return self

    def normalize(self):
        for col in self.data.select_dtypes(include=[np.float64]):
            self.data[col] = (self.data[col] - self.data[col].mean()) / math.sqrt(
                self.data[col].var()
            )

    def fill_nans(self):
        self.data.interpolate(inplace=True, method="time")
        self.data.fillna(inplace=True, method="bfill")

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

    def to_csv(self):
        self.data.to_csv(FILENAME_OUT)

    def plot(self):
        self.data.plot(y=[col for col in self.data if not col.startswith(COLUMN_TIME)])
        plt.show()


def prepare():
    # read
    preprocessor = Preprocessor.from_excel()
    # common timeline
    preprocessor.to_common_time()
    # process data
    preprocessor.normalize()
    preprocessor.sort_columns()
    preprocessor.sort_index()
    preprocessor.fill_nans()
    preprocessor.expand_columns()
    print(preprocessor.data.head(20))
    # save to file
    preprocessor.to_csv()
    # plot
    preprocessor.plot()


if __name__ == "__main__":
    prepare()
