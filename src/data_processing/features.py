import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, timedelta
import math


def convert_str_point_date_to_date(string_date):
    date_values = string_date.split('.')
    return date(year=int(date_values[2]), month=int(date_values[1]), day=int(date_values[0]))


def create_vacation_date_list(start, end):
    vacation_list = []
    while start <= end:
        vacation_list.append(start)
        start += timedelta(days=1)
    return vacation_list


def create_zurich_vaction_list():
    vacation_list_zurich = [
        create_vacation_date_list(start=date(year=2022, month=4, day=14), end=date(year=2022, month=4, day=29)),
        create_vacation_date_list(start=date(year=2022, month=7, day=18), end=date(year=2022, month=8, day=19)),
        create_vacation_date_list(start=date(year=2022, month=10, day=10), end=date(year=2022, month=10, day=21))]
    # easter
    # summer
    # autumn
    return vacation_list_zurich


def normalized_difference(a, b):
    """ return (a-b)/a if both variables are numbers"""
    if (type(a) == float or type(a) == int) and (type(b) == float or type(b) == int):
        if a == 0:
            return np.nan
        else:
            return (a - b) / a
    else:
        return np.nan


def average_of_past_weeks(num_weeks=3):
    def func(day):
        num_people_11_30 = []
        for i in range(1, num_weeks + 1):
            considered_day = day - timedelta(days=i * 7)
            try:
                num_people_11_30.append(features_by_date.loc[considered_day, 'label_num_people_11_30'])
            except KeyError:
                print(f'{considered_day} gives KeyError')
        return np.mean(num_people_11_30)

    return func


def deviation_last_workday(day_delta=1):
    def func(day):
        row_num = features_by_date.index.get_loc(day)
        try:
            deviation_last_workday = (features_by_date.iloc[row_num - day_delta]['label_num_people_11_30'] -
                                      features_by_date.iloc[row_num - day_delta]['exp_moving_avg_0.2']) / \
                                     features_by_date.iloc[row_num - day_delta]['exp_moving_avg_0.2']
        except KeyError:
            deviation_last_workday = np.nan
        return deviation_last_workday

    return func


if __name__ == '__main__':

    pickle_path = os.path.join('..', '..', 'data', 'flow_counts_preprocessed')
    df = pd.read_pickle(pickle_path)

    ##create num_people_by_date_dataframe
    date_list_all = list(df[(df['time_numeric'] == 12.33)]['date'])
    num_people_by_date = pd.DataFrame(index=date_list_all)
    num_people_by_date['date'] = num_people_by_date.index
    num_people_by_date['num_people_at_12_33'] = df.loc[df['time_numeric'] == 12.33][
        ['num_people_inside', 'date']].set_index('date')
    example_date = date(year=2022, month=5, day=23)
    num_people_by_date['num_people_at_10_00'] = df.loc[df['time_numeric'] == 10.00][
        ['num_people_inside', 'date']].set_index('date')
    num_people_by_date['num_people_at_11_30'] = df.loc[df['time_numeric'] == 11.50][
        ['num_people_inside', 'date']].set_index('date')
    num_people_by_date['normalized_difference_10_and_12'] = num_people_by_date.apply(
        lambda row: normalized_difference(row['num_people_at_10_00'], row['num_people_at_12_33']), axis=1)
    csv_path = os.path.join('..', '..', 'data', 'sold_menus.csv')
    sold_menus = pd.read_csv(csv_path, delimiter=';')
    sold_menus['Datum'] = sold_menus['Datum'].apply(convert_str_point_date_to_date)
    sold_menus.rename(columns={'Datum': 'date'}, inplace=True)
    sold_menus.set_index(keys=['date'], inplace=True)
    num_people_by_date = num_people_by_date.join(sold_menus)
    num_people_by_date['normalized_difference_10_and_sold_menus'] = num_people_by_date.apply(
        lambda row: normalized_difference(row['num_people_at_10_00'], row['sold_menus']), axis=1)
    path_to_pickle = os.path.join('..', '..', 'data', 'num_people_by_date')
    num_people_by_date.to_pickle(path_to_pickle)

    # add data of past days [] ignores past days
    relevant_num_days_ago = []  # [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    # column_names = ['label_num_people_12_33', 'label_num_menus_sold', 'num_people_10_00']
    # column_names.extend([str(i) + '_days_ago_12_33' for i in relevant_num_days_ago])
    # column_names.extend([str(i) + '_days_ago_norm_dif' for i in relevant_num_days_ago])
    features_by_date = pd.DataFrame(index=[date for date in date_list_all if date.weekday() in [0, 1, 2, 3, 4]])
    features_by_date['date'] = num_people_by_date['date']
    features_by_date['num_people_10_00'] = num_people_by_date['num_people_at_10_00']
    features_by_date['label_num_people_12_33'] = num_people_by_date['num_people_at_12_33']
    features_by_date['label_num_people_11_30'] = num_people_by_date['num_people_at_11_30']
    features_by_date['label_num_menus_sold'] = num_people_by_date['sold_menus']
    features_by_date['label_difference_12_normalized'] = num_people_by_date['normalized_difference_10_and_12']
    features_by_date['label_difference_10_and_sold_normalized'] = num_people_by_date[
        'normalized_difference_10_and_sold_menus']
    zurich_vacation_list = create_zurich_vaction_list()
    features_by_date['zurich_vacation'] = features_by_date['date'].isin(zurich_vacation_list).astype(int)

    #add exp moving average as feature
    for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        exp_mov_averages = pd.Series(index=features_by_date.date, dtype=float)
        for day in features_by_date.date:
            try:
                day_7d_ago=day-timedelta(days=7)
                avg_7d_ago = exp_mov_averages[day_7d_ago]
                new_datapoint=features_by_date.loc[day_7d_ago, 'label_num_people_11_30']
                if new_datapoint<30:
                    exp_mov_averages[day] = avg_7d_ago
                elif math.isnan(avg_7d_ago):
                    exp_mov_averages[day] = new_datapoint
                else:
                    exp_mov_averages[day] = alpha * features_by_date.loc[day_7d_ago, 'label_num_people_11_30'] + (
                            1 - alpha) * avg_7d_ago
            except KeyError:  # no measurements at start
                pass
        features_by_date[f'exp_moving_avg_{alpha}']=exp_mov_averages
        features_by_date[f'log_exp_moving_avg_{alpha}']=features_by_date.apply(lambda row: math.log(row[f'exp_moving_avg_{alpha}']), axis=1)

    # add weekday as feature
    features_by_date['weekday'] = features_by_date['date'].apply(date.weekday)
    for i, weekday in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']):
        features_by_date[weekday] = features_by_date['weekday'].apply(lambda x: int(x == i))

    # add season
    features_by_date['Spring/Autumn'] = features_by_date['date'].apply(lambda day: int(day.month in [4, 5, 9, 10]))
    features_by_date['Summer'] = features_by_date['date'].apply(lambda day: int(day.month in [6, 7, 8]))
    features_by_date['Winter'] = features_by_date['date'].apply(lambda day: int(day.month in [11, 12, 1, 2, 3]))

    # add month quarter indicators
    month_quarter_lims = [1, 9, 16, 24, 32]
    for i in range(0, len(month_quarter_lims) - 1):
        features_by_date[f'month_quarter_{i}'] = features_by_date['date'].apply(
            lambda day: int(day.day in range(month_quarter_lims[i], month_quarter_lims[i + 1])))

    # add zuhlke days
    zuehlke_days = [
        date(year=2022, month=11, day=30),
        date(year=2022, month=11, day=25), #Year end party
        date(year=2022, month=11, day=10), #Zukunftstag
        date(year=2022, month=10, day=27),
        date(year=2022, month=9, day=28),
        date(year=2022, month=8, day=31),
        date(year=2022, month=6, day=30),
        date(year=2022, month=5, day=31),
        date(year=2022, month=5, day=3),
        date(year=2022, month=3, day=30)
    ]
    features_by_date['zuehlke_day'] = features_by_date['date'].isin(zuehlke_days).astype(int)
    # add day after zuhlke day
    day_after_zuehlke_day = [ day +timedelta(days=1) for day in zuehlke_days]
    features_by_date['day_after_zuehlke_day'] = features_by_date['date'].isin(day_after_zuehlke_day).astype(int)
    # add day before zuhlke day
    day_before_zuehlke_day = [ day -timedelta(days=1) for day in zuehlke_days]
    features_by_date['day_before_zuehlke_day'] = features_by_date['date'].isin(day_before_zuehlke_day).astype(int)


    # add holydays
    holydays = [
        date(year=2022, month=4, day=15),
        date(year=2022, month=4, day=18),
        date(year=2022, month=5, day=1),
        date(year=2022, month=5, day=26),
        date(year=2022, month=5, day=27),
        date(year=2022, month=6, day=6),
        date(year=2022, month=8, day=1)
    ]
    features_by_date['holydays'] = features_by_date['date'].isin(holydays).astype(int)

    #day before and after holyday:
    before_after_holydays = [
        date(year=2022, month=4, day=14),
        date(year=2022, month=4, day=19),
        # date(year=2022, month=5, day=1), this is a sunday in 2022
        date(year=2022, month=5, day=25),
        date(year=2022, month=5, day=30),
        date(year=2022, month=6, day=3),
        date(year=2022, month=6, day=7),
        date(year=2022, month=7, day=29),
        date(year=2022, month=8, day=2)
    ]
    features_by_date['before_after_holydays'] = features_by_date['date'].isin(before_after_holydays).astype(int)

    #holydays in other cantons
    maybe_holydays=[
        date(year=2022, month=4, day=25),  # Sechseleuten?
        date(year=2022, month=9, day=12),  # Knabenschiessen?
        date(year=2022, month=6, day=16),  # Fronleichnam?
        date(year=2022, month=6, day=17),  # Fronleichnam?
        date(year=2022, month=8, day=15),  # MAria Himmelfahrt?
        date(year=2022, month=11, day=1),  # Allerheiligen?
        date(year=2022, month=12, day=8)  # Mariä Empfängnis?
    ]
    features_by_date['maybe_holydays'] = features_by_date['date'].isin(maybe_holydays).astype(int)


    # add meteo data: Temperature in C at 12, rain Minutes in the last hour at 12
    # preprocessing, getting date into the right format
    csv_path = os.path.join('..', '..', 'data', 'ugz_ogd_meteo_h1_2022.csv')
    meteo_data = pd.read_csv(csv_path)
    meteo_data['date'] = meteo_data['Datum'].apply(lambda x: x.split('T')[0])
    meteo_data['time'] = meteo_data['Datum'].apply(lambda x: x.split('T')[1])
    meteo_data['date'] = meteo_data['date'].apply(
        lambda x: date(year=int(x.split('-')[0]), month=int(x.split('-')[1]), day=int(x.split('-')[2])))

    # adding temperature as feature
    temperature_meteo_data = meteo_data[
        (meteo_data['Standort'] == 'Zch_Rosengartenstrasse') & (meteo_data['Parameter'] == 'T') & (
                meteo_data['time'] == '12:00+0100')][['date', 'Wert']].set_index('date').rename(
        columns={'Wert': 'Temperature'})
    temperature_meteo_data.fillna(method='ffill', inplace=True)
    features_by_date = features_by_date.join(temperature_meteo_data)

    # 'temp_deviation' gives temperature deviation from mean of last 10 days
    num_past_days = 10
    temparature_mean_last_days = pd.DataFrame(
        data=np.convolve(temperature_meteo_data.values.ravel(), np.ones(num_past_days) / num_past_days, mode='valid'),
        columns=['average_temperature'], index=[date(year=2022, month=1, day=num_past_days) + timedelta(days=i) for i in
                                                range(0, len(temperature_meteo_data) + 1 - num_past_days)]
    )
    temperature_meteo_data = temperature_meteo_data.join(temparature_mean_last_days)
    temperature_meteo_data['temp_deviation'] = temperature_meteo_data.apply(
        lambda row: row['Temperature'] - row['average_temperature'], axis=1)
    features_by_date = features_by_date.join(temperature_meteo_data['temp_deviation'])

    # adding rain as feature
    # 'Rain Duration' gives minutes of rain in the last hour
    rain_meteo_data = meteo_data[
        (meteo_data['Standort'] == 'Zch_Rosengartenstrasse') & (meteo_data['Parameter'] == 'RainDur') & (
                meteo_data['time'] == '12:00+0100')][['date', 'Wert']].set_index('date').rename(
        columns={'Wert': 'Rain Duration'})
    rain_agg = meteo_data[
        (meteo_data['Standort'] == 'Zch_Rosengartenstrasse') & (meteo_data['Parameter'] == 'RainDur') & (
            meteo_data['time'].isin(['12:00+0100', '11:00+0100', '10:00+0100']))][['date', 'Wert']].rename(
        columns={'Wert': 'RainDur'}).groupby('date').aggregate(np.mean)
    # 'Rain_half_discrete' gives average minutes of rain in last three houres, if this is 0 then minus the average,
    # such that total mean is zero
    rain_meteo_data['Rain_half_discrete'] = rain_agg.apply(lambda x: x['RainDur'] if x['RainDur'] > 0 else -
    np.mean(rain_meteo_data.groupby('date').aggregate(np.mean), axis=0)[0], axis=1
                                                           )
    # 'Rain_binary' gives 1 if there was rain in last three houres otherwise 0
    rain_meteo_data['Rain_binary'] = rain_agg.apply(lambda x: 1 if x['RainDur'] > 0 else 0, axis=1)
    features_by_date = features_by_date.join(rain_meteo_data)

    # Average of past weeks on same weekday
    day = date(year=2022, month=3, day=31)
    past_weeks = 3
    average_of_past_3_weeks = average_of_past_weeks(3)
    average_of_past_3_weeks(day)
    features_by_date['average_of_past_weeks'] = features_by_date['date'].apply(average_of_past_3_weeks)

    # Deviation from average num people on past days
    dev_1_day_ago = deviation_last_workday(1)
    features_by_date['deviation_1_day_ago'] = features_by_date['date'].apply(dev_1_day_ago)
    dev_2_day_ago = deviation_last_workday(2)
    features_by_date['deviation_2_day_ago'] = features_by_date['date'].apply(dev_2_day_ago)
    dev_3_day_ago = deviation_last_workday(3)
    features_by_date['deviation_3_day_ago'] = features_by_date['date'].apply(dev_3_day_ago)
    dev_4_day_ago = deviation_last_workday(4)
    features_by_date['deviation_4_day_ago'] = features_by_date['date'].apply(dev_4_day_ago)
    dev_5_day_ago = deviation_last_workday(5)
    features_by_date['deviation_5_day_ago'] = features_by_date['date'].apply(dev_5_day_ago)


    for day in date_list_all:
        for i in relevant_num_days_ago:
            time_delta = timedelta(days=i)
            try:
                features_by_date.loc[day, (str(i) + '_days_ago_12_33')] = num_people_by_date.loc[
                    day - time_delta, 'num_people_at_12_33']
            except KeyError:
                features_by_date.loc[day, str(i) + '_days_ago_12_33'] = float('nan')
            try:
                features_by_date.loc[day, str(i) + '_days_ago_norm_dif'] = num_people_by_date.loc[
                    day - time_delta, 'normalized_difference_10_and_12']
            except KeyError:
                features_by_date.loc[day, str(i) + '_days_ago_norm_dif'] = float('nan')


    #time_series data
    # fill all vacation days with the exp moving average with alpha=0.2
    features_by_date['filled_num_people_11_30']=features_by_date.apply(lambda row: row['exp_moving_avg_0.2'] if row['label_num_people_11_30']<30 else row['label_num_people_11_30'],axis=1)
    features_by_date['log_num_people_11_30']= features_by_date.apply(lambda row: math.log(row['filled_num_people_11_30']),axis=1)


    path_to_pickle = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date.to_pickle(path_to_pickle)
