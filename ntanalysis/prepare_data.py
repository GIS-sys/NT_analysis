import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from datetime import datetime
from tqdm import tqdm


FILENAME_IN = "data/raw.xlsx"
FILENAME_OUTPUT = "data/out.csv"
DEBUG_ONLY_FIRST_N_LINES = 999999
if DEBUG_ONLY_FIRST_N_LINES != None:
    print(f"WARNING only {DEBUG_ONLY_FIRST_N_LINES=} lines will be processed")
COLUMN_TRANSLATOR1 = {
    "YLNG.33.615II00257.PV":  "current",    # Сила тока электродвигателя,А
    "YLNG.33.615VI00505.PV":  "vibr_hor_1", # Вибрация подшипника насоса горизонтальная 1
    "YLNG.33.615VI00506.PV":  "vibr_hor_2", # Вибрация подшипника насоса горизонтальная 2
    "YLNG.33.615TI00513.PV":  "temp",       # Температура среды на входе насоса
    "YLNG.33.615PZI00256.PV": "pres_in",    # Давление среды на входе насоса
    "YLNG.33.615PI00295.PV":  "pres_out",   # Давление среды после насоса
    "YLNG.33.615PDI00254.PV": "diff",       # Перепад на входном стрейнере
    "YLNG.33.615ZI00270.PV":  "pos_high",   # Положение регулирующей арматуры на напоре
    "YLNG.33.615ZI00219.PV":  "pos_low",    # Положение регулирующей арматуры min-flow
    "YLNG.33.615FI00270.PV":  "consum"      # Расход
}
class SUFFIX:
    TIME = "_time"
    VAL = "_val"
    TIME_SECONDS = "_seconds"
    TIME_DAYOFWEEK = "_dow"
COLUMN_TRANSLATOR2 = {
    "Время": SUFFIX.TIME,
    "Значение": SUFFIX.VAL
}
COLUMN_TIME = "TIME"
EPOCH_TIME = datetime(1970, 1, 1)


def read_excel_gen(filename):
    book = openpyxl.load_workbook(filename=filename, read_only=True, data_only=True)
    first_sheet = book.worksheets[0]
    return first_sheet.values

def gen_to_pd(rows_generator, nrows):
    header_row_main = next(rows_generator)
    header_row_time = next(rows_generator)
    header_row = []
    for i in range(len(header_row_main)):
        x, y = header_row_main[i], header_row_time[i]
        if x == None:
            x = header_row_main[i - 1]
        header_row.append(COLUMN_TRANSLATOR1[x] + COLUMN_TRANSLATOR2[y])
    data_rows = []
    for (i, row) in tqdm(zip(range(nrows - 2), rows_generator), total=nrows - 2):
        row = [x if x != "Bad Input" and x != "Error! Maximum number of rows" else None for x in row]
        data_rows.append(row)
    return pd.DataFrame(data_rows, columns=header_row).iloc[:, 2:]

def to_common_time(data):
    # create separate tables for each feature
    columns = {}
    for col in set(data):
        if col == None or not col.endswith(SUFFIX.TIME):
            continue
        col_pref = col[:-len(SUFFIX.TIME)]
        col_time = col_pref + SUFFIX.TIME
        col_val = col_pref + SUFFIX.VAL
        columns[col_pref] = data[[col_time, col_val]]
        columns[col_pref] = columns[col_pref].rename(columns={col_time: COLUMN_TIME, col_val: col_pref})
        columns[col_pref].dropna(inplace=True, how='all')
    # join dataframes for common timeline
    result = list(columns.values())[0]
    for dataframe in tqdm(list(columns.values())[1:]):
        result = pd.merge(result, dataframe, on=COLUMN_TIME, how="outer")
    result = result.set_index(COLUMN_TIME)
    return result

def normalize(data):
    for col in data.select_dtypes(include=[np.float64]):
        data[col] = (data[col] - data[col].mean()) / math.sqrt(data[col].var())

def fill_nans(data):
    data.interpolate(inplace=True, method="time")
    data.fillna(inplace=True, method="bfill")

def sort_columns(data):
    return data.reindex(sorted(data.columns), axis=1)

def sort_index(data):
    data.sort_index(inplace=True)

def expand_columns(data):
    data[COLUMN_TIME + SUFFIX.TIME_SECONDS] = data.apply(lambda row: (row.name - EPOCH_TIME).total_seconds(), axis=1)
    data[COLUMN_TIME + SUFFIX.TIME_DAYOFWEEK] = data.apply(lambda row: row.name.weekday(), axis=1)


data_lines = read_excel_gen(FILENAME_IN)
raw_data = gen_to_pd(data_lines, DEBUG_ONLY_FIRST_N_LINES)
full_data = to_common_time(raw_data)
normalize(full_data)
full_data = sort_columns(full_data)
sort_index(full_data)
fill_nans(full_data)
expand_columns(full_data)
print(full_data.head(20))

# save to file
full_data.to_csv(FILENAME_OUTPUT)

# data = data.drop("pres_out_val", axis=1)
full_data.plot(y=[col for col in full_data if not col.startswith(COLUMN_TIME)])
plt.show()

