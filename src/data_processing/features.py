import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, timedelta


def convert_str_point_date_to_date(string_date):
    date_values = string_date.split('.')
    return date(year=int(date_values[2]), month=int(date_values[1]), day=int(date_values[0]))

if __name__ == '__main__':

    pickle_path = os.path.join('..', '..', 'data', 'flow_counts_preprocessed')
    df = pd.read_pickle(pickle_path)


    date_list_all = list(df[(df['time_numeric'] == 12.33)]['date'])


    num_people_by_date = pd.DataFrame(index=date_list_all)
    num_people_by_date['num_people_at_12_33'] = df.loc[df['time_numeric'] == 12.33][
        ['num_people_inside', 'date']].set_index('date')
    example_date = date(year=2022, month=5, day=23)
    num_people_by_date['num_people_at_10_00'] = df.loc[df['time_numeric'] == 10.00][
        ['num_people_inside', 'date']].set_index('date')
    num_people_by_date['difference_10_and_12'] = num_people_by_date.apply(
        lambda row: row['num_people_at_10_00'] - row['num_people_at_12_33'], axis=1)
    num_people_by_date['normalized_difference_10_and_12'] = num_people_by_date.apply(
        lambda row: row['difference_10_and_12'] / row['num_people_at_10_00'], axis=1)

    csv_path = os.path.join('..','..', 'data', 'sold_menus.csv')
    sold_menus = pd.read_csv(csv_path, delimiter=';')
    sold_menus['Datum'] = sold_menus['Datum'].apply(convert_str_point_date_to_date)
    sold_menus.rename(columns={'Datum': 'date'}, inplace=True)
    sold_menus.set_index(keys=['date'], inplace=True)
    num_people_by_date = num_people_by_date.join(sold_menus)
    num_people_by_date['difference_10_and_sold_menus'] = num_people_by_date.apply(
        lambda row: row['num_people_at_10_00'] - row['sold_menus'], axis=1)
    num_people_by_date['normalized_difference_10_and_sold_menus'] = num_people_by_date.apply(
        lambda row: row['difference_10_and_sold_menus'] / row['num_people_at_10_00'], axis=1)




    path_to_pickle = os.path.join('..', '..', 'data', 'num_people_by_date')
    num_people_by_date.to_pickle(path_to_pickle)


    relevant_num_days_ago =  [1]#[1, 2, 3, 4, 5, 6, 7, 14, 21, 28]
    column_names = ['label_num_people_12_33','num_people_10_00']
    column_names.extend([str(i) + '_days_ago_12_33' for i in relevant_num_days_ago])
    column_names.extend([str(i) + '_days_ago_norm_dif' for i in relevant_num_days_ago])
    features_by_date = pd.DataFrame(index=date_list_all, columns=column_names)

    features_by_date['num_people_10_00'] = num_people_by_date['num_people_at_10_00']
    features_by_date['label_num_people_12_33']=num_people_by_date['num_people_at_12_33']

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


    path_to_pickle = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date.to_pickle(path_to_pickle)


