import pandas as pd
import numpy as np
import os
import pickle


from datetime import date, time, datetime, timedelta


def string_to_date_time(string_datetime):
    # string_datetime=df.datetime.iloc[0]
    [string_date, string_time] = string_datetime[:-1].split('T')
    [string_hour, string_minute] = string_time.split(':')
    [string_year, string_month, string_day] = string_date.split('-')
    date_time = datetime(year=int(string_year), month=int(string_month), day=int(string_day), hour=int(string_hour),
                         minute=int(string_minute))
    # date1 = date(year=int(string_year), month=int(string_month), day=int(string_day))
    # time1 = time(hour=int(string_hour), minute=int(string_minute))
    # return date1, time1, time_numeric
    return date_time


def time_to_numeric(time_object):
    return np.round(time_object.hour + time_object.minute / 60, 2)


def decide_whether_summer_time(datetime_object):
    if datetime_object.month < 3:
        return False
    elif datetime_object.month == 3:
        if datetime_object.day + 6 - datetime_object.weekday() <= 31:
            return False
        else:
            return True
    elif 3 < datetime_object.month < 10:
        return True
    elif datetime_object.month == 10:
        if datetime_object.day + 6 - datetime_object.weekday() <= 31:
            return True
        else:
            return False
    elif datetime_object.month > 10:
        return False


def convert_to_zurich_time(datetime_object):
    if decide_whether_summer_time(datetime_object):
        return datetime_object + timedelta(hours=2)
    else:
        return datetime_object + timedelta(hours=1)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == '__main__':
    csv_path = os.path.join('..', '..', 'data', 'flow_counts.csv')
    df = pd.read_csv(csv_path)
    df['datetime'] = df.apply(lambda row: string_to_date_time(row['datetime']), axis=1)
    df['zurich_datetime'] = df.apply(lambda row: convert_to_zurich_time(row['datetime']), axis=1)
    smoothing_conv_param = 3
    df['smooth_visitor_count_fw'] = smooth(df.visitor_count_fw, smoothing_conv_param)
    df['smooth_visitor_count_bw'] = smooth(df.visitor_count_bw, smoothing_conv_param)
    df['date'] = df.apply(lambda row: row['zurich_datetime'].date(), axis=1)
    df['time'] = df.apply(lambda row: row['zurich_datetime'].time(), axis=1)
    df['time_numeric'] = df.apply(lambda row: time_to_numeric(row['time']), axis=1)
    df['cum_sum_visitor_count_fw'] = df[['date', 'visitor_count_fw']].groupby('date').cumsum(axis=0)
    df['cum_sum_visitor_count_bw'] = df[['date', 'visitor_count_bw']].groupby('date').cumsum(axis=0)
    df['num_people_inside'] = df.apply(lambda row: row['cum_sum_visitor_count_fw'] - row['cum_sum_visitor_count_bw'],
                                       axis=1)
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]
    df['weekday'] = df.apply(lambda row: days[row['date'].weekday()], axis=1)
    df['weekday_num'] = df.apply(lambda row: row['date'].weekday(), axis=1)
    csv_file_name = 'flow_counts_preprocessed.csv'
    path_to_master_csv = os.path.join('..', '..', 'data', csv_file_name)
    df.to_csv(path_to_master_csv, index=False)
    pickle_file_name = 'flow_counts_preprocessed'
    path_to_master_pickle = os.path.join('..', '..', 'data', pickle_file_name)
    df.to_pickle(path_to_master_pickle)
    print(f'wrote csv file to: {path_to_master_csv}')

