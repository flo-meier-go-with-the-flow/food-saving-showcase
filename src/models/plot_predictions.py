import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from pprint import pprint

import json
import pickle


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('..', '..', 'reports', 'figures', fig_id + "." + fig_extension)
    print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


if __name__ == '__main__':

    label_0 = 'label_num_people_11_30'
    with open(os.path.join('predictions', 'x_data_' + label_0 + '.pkl'), 'rb') as file:
        x_data_11_30 = pickle.load(file)
    with open(os.path.join('predictions', 'y_data_' + label_0 + '.pkl'), 'rb') as file:
        y_data_11_30 = pickle.load(file)
    with open(os.path.join('predictions', 'y_predictions_' + label_0 + '.pkl'), 'rb') as file:
        y_predictions_11_30 = pickle.load(file)
    label_1 = 'label_num_people_12_33'
    with open(os.path.join('predictions', 'x_data_' + label_1 + '.pkl'), 'rb') as file:
        x_data_12_33 = pickle.load(file)
    with open(os.path.join('predictions', 'y_data_' + label_1 + '.pkl'), 'rb') as file:
        y_data_12_33 = pickle.load(file)
    with open(os.path.join('predictions', 'y_predictions_' + label_1 + '.pkl'), 'rb') as file:
        y_predictions_12_33 = pickle.load(file)
    label_2 = 'label_num_menus_sold'
    with open(os.path.join('predictions', 'x_data_' + label_2 + '.pkl'), 'rb') as file:
        x_data_menus_sold = pickle.load(file)
    with open(os.path.join('predictions', 'y_data_' + label_2 + '.pkl'), 'rb') as file:
        y_data_menus_sold = pickle.load(file)
    with open(os.path.join('predictions', 'y_predictions_' + label_2 + '.pkl'), 'rb') as file:
        y_predictions_menus_sold = pickle.load(file)

    y_predictions_11_30 = pd.Series(data=y_predictions_11_30, index=x_data_11_30.index)
    y_predictions_12_33 = pd.Series(data=y_predictions_12_33, index=x_data_12_33.index)
    y_predictions_menus_sold = pd.Series(data=y_predictions_menus_sold, index=x_data_menus_sold.index)

    plot_first_error_visualization = True
    if plot_first_error_visualization:
        start_day = date(year=2022, month=11, day=14)
        for i in range(0, 1):
            day = start_day + timedelta(days=i)

            num_people_10_00 = x_data_11_30.loc[day, 'num_people_10_00']
            num_people_11_30_prediction = y_predictions_11_30.loc[day]
            num_people_11_30_data = y_data_11_30.loc[day, 'label_num_people_11_30']
            num_people_12_33_prediction = y_predictions_12_33.loc[day]
            num_people_12_33_data = y_data_12_33.loc[day, 'label_num_people_12_33']
            menus_sold_prediction = y_predictions_menus_sold.loc[day]
            menus_sold_data = y_data_menus_sold.loc[day, 'label_num_menus_sold']
            #plt.plot([10, 11.5, 12.33, 14],
            #         [num_people_10_00, num_people_11_30_data, num_people_12_33_data, menus_sold_data], 'ro')
            plt.plot([10],
                     [num_people_10_00], 'ro')

            plt.errorbar([11.5, 12.33, 14],
                         [num_people_11_30_prediction, num_people_12_33_prediction, menus_sold_prediction],
                         yerr=[4.1, 6.7, 14.8], capsize=5.0, fmt='bo')
            plt.legend(['data', 'predictions'])
            plt.ylim(0, 300)
            # plt.annotate('11:30, av. Fehler: 4.1', xy=(11.5, num_people_11_30_prediction), xytext=(8, 250),
            #            arrowprops=dict(facecolor='black', shrink=0.02))
            plt.text(13.1, 123, r'Verkaufte Menus', fontsize=9)
            plt.text(13.4, 174, r'Error 14.8', fontsize=9, color='green')
            # plt.text(12.5, 178, r'Anzahl Pers 12:20 :', fontsize=9)
            plt.text(11.9, 195, r'Error 6.7', fontsize=9, color='green')
            # plt.text(10.5, 228, r'Anzahl Pers 11:30 :', fontsize=9)
            plt.text(11.0, 212, r'Error 4.1', fontsize=9, color='green')
            plt.xlim(8, 18)
            plt.xlabel('time')
            plt.ylabel('number people')
            plt.xticks([10, 11, 12, 13])
            plt.grid(True)
            fig_title = '0_Predictions_' + str(day)
            plt.title(fig_title)
            fig_id = '0_Predictions_without_data' + str(day)
            save_fig(fig_id)
            plt.clf()

    plot_errors_subplots = False
    if plot_errors_subplots:
        start_day = date(year=2022, month=11, day=15)
        for i_num, j in enumerate([0, 1, 2, 3, 6, 7, 8, 9]):
            day = start_day + timedelta(days=j)

            num_people_10_00 = x_data_11_30.loc[day, 'num_people_10_00']
            num_people_11_30_prediction = y_predictions_11_30.loc[day]
            num_people_11_30_data = y_data_11_30.loc[day, 'label_num_people_11_30']
            num_people_12_33_prediction = y_predictions_12_33.loc[day]
            num_people_12_33_data = y_data_12_33.loc[day, 'label_num_people_12_33']
            menus_sold_prediction = y_predictions_menus_sold.loc[day]
            menus_sold_data = y_data_menus_sold.loc[day, 'label_num_menus_sold']
            subplotnum = i_num + 241
            plt.subplot(subplotnum)
            plt.plot([10, 11.5, 12.33, 14],
                     [num_people_10_00, num_people_11_30_data, num_people_12_33_data, menus_sold_data], 'ro')
            plt.errorbar([11.5, 12.33, 14],
                         [num_people_11_30_prediction, num_people_12_33_prediction, menus_sold_prediction],
                         yerr=[4.1, 6.7, 14.8], capsize=5.0, fmt='bo')
            plt.ylim(50, 260)
            plt.xlim(9.5, 14.5)
            plt.xticks([10, 11, 12, 13])
            plt.title(str(day))
            plt.grid(True)
        fig_id = '0_Predictions_subplots_' + str(start_day)
        save_fig(fig_id)

        plt.clf()

    plot_errors_zuehlke_days = False
    if plot_errors_zuehlke_days:
        zuehlke_days = [
            # date(year=2022, month=11, day=30),
            date(year=2022, month=10, day=27),
            date(year=2022, month=9, day=28),
            date(year=2022, month=8, day=31),
            date(year=2022, month=6, day=30),
            date(year=2022, month=5, day=31),
            date(year=2022, month=5, day=3),
            date(year=2022, month=3, day=30)
        ]
        for i_num, day in enumerate(zuehlke_days):
            num_people_10_00 = x_data_11_30.loc[day, 'num_people_10_00']
            num_people_11_30_prediction = y_predictions_11_30.loc[day]
            num_people_11_30_data = y_data_11_30.loc[day, 'label_num_people_11_30']
            num_people_12_33_prediction = y_predictions_12_33.loc[day]
            num_people_12_33_data = y_data_12_33.loc[day, 'label_num_people_12_33']
            menus_sold_prediction = y_predictions_menus_sold.loc[day]
            menus_sold_data = y_data_menus_sold.loc[day, 'label_num_menus_sold']
            subplotnum = i_num + 241
            plt.subplot(subplotnum)
            plt.plot([10, 11.5, 12.33, 14],
                     [num_people_10_00, num_people_11_30_data, num_people_12_33_data, menus_sold_data], 'ro')
            plt.errorbar([11.5, 12.33, 14],
                         [num_people_11_30_prediction, num_people_12_33_prediction, menus_sold_prediction],
                         yerr=[4.1, 6.7, 14.8], capsize=5.0, fmt='bo')
            plt.ylim(100, 300)
            plt.xlim(9.5, 14.5)
            plt.xticks([10, 11, 12, 13])
            plt.grid(True)
            plt.title(str(day))

        fig_id = '0_Predictions_zuehlke_days'
        save_fig(fig_id)
        plt.clf()

    plot_data_per_month = False
    if plot_data_per_month:
        y_data_11_30['date'] = y_data_11_30.index
        y_data_11_30['month'] = y_data_11_30.apply(lambda row: row['date'].month, axis=1)
        y_data_11_30['day'] = y_data_11_30.apply(lambda row: row['date'].day, axis=1)
        y_data_11_30 = y_data_11_30.join(y_data_12_33)
        y_data_11_30 = y_data_11_30.join(y_data_menus_sold)

        for group, frame in y_data_11_30.groupby('month'):
            consider_month = 3
            if group >= consider_month:
                subplot_num = 331 + group - consider_month
                plt.subplot(subplot_num)
                df_to_plot = frame[['day', 'label_num_people_11_30']].set_index('day')
                plt.plot(df_to_plot)
                df_to_plot = frame[['day', 'label_num_people_12_33']].set_index('day')
                plt.plot(df_to_plot)
                df_to_plot = frame[['day', 'label_num_menus_sold']].set_index('day')
                plt.plot(df_to_plot)
                plt.ylim(30, 300)
                plt.xlim(1, 31)
                plt.xticks([10, 20, 30])
                plt.title(f'Month={group}')
                plt.grid(True)
                if group == 3:
                    plt.legend(['Personen 11:30', 'Personen 12:20', 'verkaufte Menus'], loc="upper left")
        fig_id = '0_data_per_month'
        save_fig(fig_id)
        plt.clf()

    plot_data_sept = False
    if plot_data_sept:
        y_data_11_30['date'] = y_data_11_30.index
        y_data_11_30['month'] = y_data_11_30.apply(lambda row: row['date'].month, axis=1)
        y_data_11_30['day'] = y_data_11_30.apply(lambda row: row['date'].day, axis=1)
        y_data_11_30 = y_data_11_30.join(y_data_12_33)
        y_data_11_30 = y_data_11_30.join(y_data_menus_sold)
        y_data_11_30 = y_data_11_30.join(x_data_11_30)
        for group, frame in y_data_11_30.groupby('month'):

            if group == 9:
                df_to_plot = frame[['day', 'label_num_people_11_30']].set_index('day')
                plt.plot(df_to_plot)
                df_to_plot = frame[['day', 'num_people_10_00']].set_index('day')
                plt.plot(df_to_plot)
                df_to_plot = frame[['day', 'label_num_people_12_33']].set_index('day')
                plt.plot(df_to_plot)
                df_to_plot = frame[['day', 'label_num_menus_sold']].set_index('day')
                plt.plot(df_to_plot)
                plt.ylim(30, 300)
                plt.xlim(1, 31)
                plt.xticks([10, 20, 30])
                plt.xlabel('day in month')
                plt.ylabel('number people')
                plt.title(f'Month={group}')
                plt.grid(True)

                plt.legend(['Personen 11:30', 'Personen 10:00', 'Personen 12:20', 'verkaufte Menus'], loc="upper left")
        fig_id = '0_data_september'
        save_fig(fig_id)
        plt.clf()
    print('debug')
