from datetime import date,timedelta
import os
import pandas as pd
import pickle


def day_is_zuehlke_day(example_date):
    zuehlke_days = [
        date(year=2023, month=11, day=29),
        date(year=2023, month=11, day=2),
        date(year=2023, month=9, day=28),
        date(year=2023, month=8, day=29),
        date(year=2023, month=6, day=29),
        date(year=2023, month=5, day=30),
        date(year=2023, month=4, day=27),
        date(year=2023, month=3, day=30),
        date(year=2023, month=2, day=28),
        date(year=2023, month=1, day=26)
    ]
    return int(example_date in zuehlke_days)

def day_is_before_after_holydays(example_date):
    before_after_holydays = [
        date(year=2023, month=1, day=2),
        date(year=2023, month=4, day=6),
        date(year=2023, month=4, day=11),
        date(year=2023, month=4, day=28),
        date(year=2023, month=5, day=2),
        date(year=2023, month=5, day=17),
        date(year=2023, month=5, day=22),
        date(year=2023, month=5, day=26),
        date(year=2023, month=5, day=30),
        date(year=2023, month=7, day=31),
        date(year=2023, month=8, day=2),
        date(year=2023,month=12,day=22)
    ]
    return int(example_date in before_after_holydays)


if __name__ == '__main__':

    pickle_path = os.path.join('features.pkl')
    features = pd.read_pickle(pickle_path)


    this_date=date(year=2023,month=1,day=1)
    for i in range(365):
        this_date+=timedelta(days=1)
        if this_date.weekday() in [0,1,2,3,4]:
            features.loc[this_date,'zuehlke_day']=day_is_zuehlke_day(this_date)
            features.loc[this_date,'before_after_holydays']=day_is_before_after_holydays(this_date)
    with open('features_2023.pkl', 'wb') as file:
        pickle.dump(features, file)



