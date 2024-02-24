import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


FILENAME = "data/raw.csv"
FILENAME_OUTPUT = "data/out.csv"
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
TIME_SUFFIX, VAL_SUFFIX = "_time", "_val"
TIME_SECONDS_SUFFIX, TIME_DAYOFWEEK_SUFFIX = "_seconds", "_dow"
COLUMN_TRANSLATOR2 = {
    "Время": TIME_SUFFIX,
    "Значение": VAL_SUFFIX
}
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
EPOCH_TIME = datetime(1970, 1, 1)
DEBUG_START_FROM_N_LINE = None
DEBUG_ONLY_FIRST_N_LINES = 500000

if DEBUG_START_FROM_N_LINE != None:
    print(f"WARNING first {DEBUG_START_FROM_N_LINE=} lines will be skipped")
if DEBUG_ONLY_FIRST_N_LINES != None:
    print(f"WARNING only {DEBUG_ONLY_FIRST_N_LINES=} lines will be processed")

def load_data():
    column_names = []
    data = {}
    with open(FILENAME, "r") as f:
        csv_data = csv.reader(f)
        ignore_columns = set()
        for i, row in tqdm(enumerate(csv_data)):
            if i == 0:
                for raw_column in row[::2]:
                    column_names += [COLUMN_TRANSLATOR1[raw_column]] * 2
            elif i == 1:
                for i, suffix in enumerate(row):
                    column_names[i] += COLUMN_TRANSLATOR2[suffix]
                for col_ind, col_name in enumerate(column_names):
                    if col_name in column_names[col_ind+1:]:
                        ignore_columns.add(col_ind)
                    else:
                        data[col_name] = []
            else:
                if DEBUG_START_FROM_N_LINE != None and i < DEBUG_START_FROM_N_LINE:
                    continue
                if DEBUG_START_FROM_N_LINE != None:
                    if DEBUG_ONLY_FIRST_N_LINES != None and i > DEBUG_ONLY_FIRST_N_LINES + DEBUG_START_FROM_N_LINE:
                        break
                else:
                    if DEBUG_ONLY_FIRST_N_LINES != None and i > DEBUG_ONLY_FIRST_N_LINES:
                        break
                for col_ind, (col_name, cell) in enumerate(zip(column_names, row)):
                    if col_ind in ignore_columns:
                        continue
                    data[col_name].append(cell)
    return data

def parse_float(s):
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None

def parse_date(s):
    try:
        return datetime.strptime(s, DATETIME_FORMAT)
    except ValueError:
        pass
    return datetime.strptime(s + " 00:00:00", DATETIME_FORMAT)

def parse_date_to_sec(s):
    try:
        return int((parse_date(s) - EPOCH_TIME).total_seconds())
    except ValueError:
        return None

def parse_date_to_dow(s):
    try:
        return parse_date(s).weekday()
    except ValueError:
        return None

def parse_data(data):
    result = {}
    for col_name, list_of_vals in tqdm(data.items()):
        result[col_name] = []
        if col_name.endswith(TIME_SUFFIX):
            result[col_name + TIME_SECONDS_SUFFIX] = [] # column for time in seconds
            # result[col_name + TIME_DAYOFWEEK_SUFFIX] = [] # column for day of week # TODO return this
            for val in list_of_vals:
                result[col_name].append(val)
                result[col_name + TIME_SECONDS_SUFFIX].append(parse_date_to_sec(val))
                # result[col_name + TIME_DAYOFWEEK_SUFFIX].append(parse_date_to_dow(val)) # TODO return this
        elif col_name.endswith(VAL_SUFFIX):
            for val in list_of_vals:
                result[col_name].append(parse_float(val))
        else:
            raise Exception("parse_data(...) got column with an unexpected suffix")
    return result


data = load_data()
data = parse_data(data)
data = pd.DataFrame(data).iloc[:-1]
#data = data.drop('pos_low_time', axis=1).drop('pos_low_val', axis=1).drop('diff_time', axis=1).drop('diff_val', axis=1) # TODO these columns have huge amounts of NaN
#for x in ["current_time", "current_val", "vibr_hor_1_time", "vibr_hor_1_val", "vibr_hor_2_time", "vibr_hor_2_val", "temp_time", "temp_val", "pres_in_time", "pres_in_val", "pres_out_time", "pres_out_val"]: # good columns
#    data = data.drop(x, axis=1)
#print(data[data.isna().any(axis=1)])
#print(data)


print(data)
if True:
    # PREPROCESSING plot with time in seconds and mean
    # find min and max time
    TIME_MIN, TIME_MAX = 1000000000000, -1
    for col in data:
        if col.endswith(TIME_SECONDS_SUFFIX):
            TIME_MIN = min(TIME_MIN, data[col].min())
            TIME_MAX = max(TIME_MAX, data[col].max() + 1)
    TIME_MIN, TIME_MAX = int(TIME_MIN), int(TIME_MAX)
    print(f"{TIME_MIN=}, {TIME_MAX=}")
    # create table for times and Nones else
    processed_data = {"TIME": [0] + [i for i in range(int(TIME_MIN), int(TIME_MAX))]}
    for col in data:
        if col.endswith(VAL_SUFFIX):
            processed_data[col] = [0] + [None for i in range(int(TIME_MIN), int(TIME_MAX))]
    for i, col in enumerate(processed_data):
        if col == "TIME":
            continue
        col_seconds = col[:col.index(VAL_SUFFIX)] + TIME_SUFFIX + TIME_SECONDS_SUFFIX
        mean = data[col].mean() / (1 + i)
        for cur_sec, cur_val in tqdm(zip(data[col_seconds], data[col])):
            processed_data[col][int(cur_sec) - TIME_MIN + 1] = cur_val / mean
    # remove lines with full nones
    processed_data_no_full_none = {k: [] for k in processed_data}
    last_notnone_vals = {k: None for k in processed_data if k != "TIME"}
    for i in range(len(processed_data["TIME"])):
        vals = {k: processed_data[k][i] for k in processed_data if k != "TIME"}
        all_none = True
        for k, v in vals.items():
            if not (v is None):
                all_none = False
                last_notnone_vals[k] = v
        if not all_none:
            processed_data_no_full_none["TIME"].append(i)
            for k, v in vals.items():
                if v is None:
                    v = last_notnone_vals[k]
                processed_data_no_full_none[k].append(v)
    data = pd.DataFrame(processed_data_no_full_none).iloc[1:]
    print(data)
else:
    for col in data:
        if not col.endswith(VAL_SUFFIX):
            data = data.drop(col, axis=1)
    data = data.assign(TIME=[i for i in range(len(data))])

# save to file
data.to_csv(FILENAME_OUTPUT)

# data = data.drop("pres_out_val", axis=1)
data.plot(x="TIME")
plt.show()

