import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date, time, datetime




def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('..','..','reports','figures', fig_id + "." + fig_extension)
    print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def date_string_to_weekday(date_string):
    [year, month, day]=date_string.split('-')
    days = ["Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"]
    return days[date(year=int(year), month=int(month), day=int(day)).weekday()]

if __name__ == '__main__':


    csv_path=os.path.join('..','..', 'data', 'flow_counts_preprocessed.csv')
    df= pd.read_csv(csv_path)

    df_check=df['date']

    by_date=df.groupby(by='date')

    for day_date, day_frame in by_date:
        #plot flow_counts
        df_fw_count= day_frame.set_index('time_numeric')['smooth_visitor_count_fw']
        df_bw_count= day_frame.set_index('time_numeric')['smooth_visitor_count_bw']
        plt.plot(df_fw_count)
        plt.plot(df_bw_count)
        plt.legend(['incoming_count', 'outgoing_count'])

        fig_title=day_date+' '+ date_string_to_weekday(day_date)
        plt.title(fig_title)
        fig_id='fw_bw_count_on_'+day_date
        save_fig(fig_id)
        plt.clf()

        #plot num people inside

        df_fw_count_cum_sum=day_frame.set_index('time_numeric')['cum_sum_visitor_count_fw']
        df_bw_count_cum_sum=day_frame.set_index('time_numeric')['cum_sum_visitor_count_bw']
        df_num_people_inside=day_frame.set_index('time_numeric')['num_people_inside']
        plt.plot(df_fw_count_cum_sum)
        plt.plot(df_bw_count_cum_sum)
        plt.plot(df_num_people_inside)
        plt.legend(['fw_count_sum', 'bw_count_sum', 'num_people_inside'])

        fig_title=day_date+' '+ date_string_to_weekday(day_date)
        plt.title(fig_title)
        fig_id = 'num_people_inside_on_' + day_date
        save_fig(fig_id)
        plt.clf()




