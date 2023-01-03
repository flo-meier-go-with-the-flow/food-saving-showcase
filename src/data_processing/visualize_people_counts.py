import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, time, datetime


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('..', '..', 'reports', 'figures', fig_id + "." + fig_extension)
    print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def date_string_to_weekday(date_string):
    [year, month, day] = date_string.split('-')
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]
    return days[date(year=int(year), month=int(month), day=int(day)).weekday()]


if __name__ == '__main__':
    plot_by_dates = False
    plot_quantiles_per_weekday=False
    plot_overview_week=False
    plot_overview_scatter_year=False
    plot_weekday_samples=False

    csv_path = os.path.join('..', '..', 'data', 'flow_counts_preprocessed.csv')
    df = pd.read_csv(csv_path)

    #plot for each day 1) fw bw counts, 2) num people inside
    if plot_by_dates:
        by_date = df.groupby(by='date')

        for day_date, day_frame in by_date:
            # plot flow_counts
            df_fw_count = day_frame.set_index('time_numeric')['smooth_visitor_count_fw']
            df_bw_count = day_frame.set_index('time_numeric')['smooth_visitor_count_bw']
            plt.plot(df_fw_count)
            plt.plot(df_bw_count)
            plt.legend(['incoming_count', 'outgoing_count'])

            fig_title = day_date + ' ' + date_string_to_weekday(day_date)
            plt.title(fig_title)
            fig_id = 'fw_bw_count_on_' + day_date
            save_fig(fig_id)
            plt.clf()

            # plot num people inside

            df_fw_count_cum_sum = day_frame.set_index('time_numeric')['cum_sum_visitor_count_fw']
            df_bw_count_cum_sum = day_frame.set_index('time_numeric')['cum_sum_visitor_count_bw']
            df_num_people_inside = day_frame.set_index('time_numeric')['num_people_inside']
            plt.plot(df_fw_count_cum_sum)
            plt.plot(df_bw_count_cum_sum)
            plt.plot(df_num_people_inside)
            plt.legend(['fw_count_sum', 'bw_count_sum', 'num_people_inside'])

            fig_title = day_date + ' ' + date_string_to_weekday(day_date)
            plt.title(fig_title)
            fig_id = 'num_people_inside_on_' + day_date
            save_fig(fig_id)
            plt.clf()

    # plot mean and quantiles for every weekday
    plot_quantiles_per_weekday=True
    if plot_quantiles_per_weekday:
        day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        for weekday in day_list:

            mean_values = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate([np.mean])
            std_values = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate([np.std])
            quantile_90 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,
                                                                                                           q=90)
            quantile_10 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,
                                                                                                           q=10)
            quantile_75 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,
                                                                                                           q=75)
            quantile_25 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,
                                                                                                           q=25)
            plt.plot(mean_values)
            plt.plot(quantile_90)
            plt.plot(quantile_10)
            plt.plot(quantile_75)
            plt.plot(quantile_25)
            plt.legend(['mean_values', 'quantile_90', 'quantile_10', 'quantile_75', 'quantile_25'])
            plt.ylim(0,300)
            plt.axvline(x=12.33, ymin=0, ymax=300,linewidth=0.5, label='12:20') #12:20 is (empirically found) the time with minimal num people in office
            plt.axvline(x=11.5, ymin=0, ymax=300, linewidth=0.5, label='11:30')
            plt.axvline(x=10.0, ymin=0, ymax=300,linewidth=0.5, label='10:00') #10:00 is the time mensa needs to know the number of meals to cook
            plt.xlim(0,24)
            plt.xticks([4,6,8,10,11,12,13,14,16,18,20,22])
            plt.xlabel('time')
            plt.ylabel('number people')
            fig_title = 'Average num people inside on ' + weekday
            plt.title(fig_title)
            fig_id = 'average_num_people_inside_on_' + weekday
            save_fig(fig_id)
            plt.clf()

    #plot weekday samples
    plot_weekday_samples=True
    if plot_weekday_samples:
        day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        for weekday in day_list:
            mean_values = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(
                [np.mean])
            # quantile_90 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,q=90)
            # quantile_10 = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate(np.percentile,q=10)
            plt.plot(mean_values)
            # plt.plot(quantile_90)
            # plt.plot(quantile_10)
            plt.legend(['mean_values'])
            samples = list(df[df['weekday'].isin([weekday])]['date'].sample(n=7, random_state=42))
            for sample in samples:
                df_to_plot = df[(df['weekday'].isin([weekday])) & (df['date'] == sample)][
                    ['time_numeric', 'num_people_inside']].set_index('time_numeric')
                plt.plot(df_to_plot)

            plt.ylim(0,300)
            plt.axvline(x=12.33, ymin=0, ymax=300,linewidth=0.5, label='12:20') #12:20 is (empirically found) the time with minimal num people in office
            plt.axvline(x=11.5, ymin=0, ymax=300,linewidth=0.5, label='11:30') #10:00 is the time mensa needs to know the number of meals to cook
            plt.axvline(x=10.0, ymin=0, ymax=300,linewidth=0.5, label='10:00') #10:00 is the time mensa needs to know the number of meals to cook
            plt.xlim(0,24)
            plt.xticks([4,6,8,10,11,12,13,14,16,18,20,22])
            plt.xlabel('time')
            plt.ylabel('number people')
            fig_title = 'Sample days ' + weekday
            plt.title(fig_title)
            fig_id = '0_sample_days_' + weekday
            save_fig(fig_id)
            plt.clf()


    # plot overview week
    plot_overview_week=True
    if plot_overview_week:
        day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        for weekday in day_list:

            mean_values = df[df['weekday'].isin([weekday])].groupby('time_numeric')['num_people_inside'].aggregate([np.mean])
            plt.plot(mean_values)

        plt.legend(day_list)
        plt.ylim(0,300)
        plt.axvline(x=12.33, ymin=0, ymax=300,linewidth=0.5, label='12:20') #12:20 is (empirically found) the time with minimal num people in office
        plt.axvline(x=11.50, ymin=0, ymax=300,linewidth=0.5, label='11:30') #max morning
        plt.axvline(x=10.0, ymin=0, ymax=300,linewidth=0.5, label='10:00') #10:00 is the time mensa needs to know the number of meals to cook
        plt.annotate('10:00', xy=(10.0, 290), xytext=(4, 285),
                    arrowprops=dict(facecolor='black',width=0.5,headwidth=3,headlength=3, shrink=0.0))
        plt.annotate('11:30', xy=(11.5, 270), xytext=(4, 265),
                    arrowprops=dict(facecolor='black',width=0.5,headwidth=3,headlength=3, shrink=0.0))
        plt.annotate('12:20', xy=(12.33, 250), xytext=(4, 245),
                    arrowprops=dict(facecolor='black',width=0.5,headwidth=3,headlength=3, shrink=0.0))
        plt.xlim(0,24)
        plt.xticks([4,6,8,10,11,12,13,14,16,18,20,22])
        plt.xlabel('time')
        plt.ylabel('number people')
        fig_title = 'Average num people inside'
        plt.title(fig_title)
        fig_id = '0_average_num_people_inside'
        save_fig(fig_id)
        plt.clf()




    #plot num people inside trajectories
    if plot_overview_scatter_year:
        pickle_path = os.path.join('..', '..', 'data', 'num_people_by_date')
        num_people_by_date = pd.read_pickle(pickle_path)
        num_people_by_date['date'] = num_people_by_date.index
        num_people_by_date['weekday'] = num_people_by_date['date'].apply(date.weekday)
        num_people_workdays=num_people_by_date[num_people_by_date['num_people_at_12_33']>30]

        for i in [0,1,2,3,4]:
            df_to_plot=num_people_workdays[num_people_workdays['weekday']==i]
            plt.scatter(df_to_plot.index,df_to_plot['num_people_at_12_33'])
        plt.legend(['Mon', 'Tue', 'Wed','Thu','Fri'])

        fig_title = 'num people inside at 12:20'
        plt.title(fig_title)
        fig_id = '0_overview_num_people_inside_12_33'
        save_fig(fig_id)
        plt.clf()

        for i in [0,1,2,3,4]:
            df_to_plot=num_people_workdays[num_people_workdays['weekday']==i]
            plt.scatter(df_to_plot.index,df_to_plot['normalized_difference_10_and_12'])
        plt.legend(['Mon', 'Tue', 'Wed','Thu','Fri'])

        fig_title = '(numpeople_at_10 - num_people_at_12)/numpeople_at_10'
        plt.title(fig_title)
        fig_id = '0_overview_normalized_difference_10_and_12'
        save_fig(fig_id)
        plt.clf()


