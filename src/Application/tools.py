import os
import pandas as pd
from src.models.sarimax_prediction_future_days import train_sarimax_mdoel_for_prediction
from datetime import date, datetime, time,timedelta
import math
import pickle
import numpy as np



def is_public_holyday(date_object):
    holydays = [
        date(year=2023, month=1, day=1),
        date(year=2023, month=4, day=7),
        date(year=2023, month=4, day=10),
        date(year=2023, month=5, day=1),
        date(year=2023, month=5, day=18),
        date(year=2023, month=5, day=19),
        date(year=2023, month=5, day=29),
        date(year=2023, month=8, day=1)
    ]
    return False

def display_day_time():
    weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunay']
    date_time_now=datetime.now()
    weekday= weekdays[date_time_now.weekday()]
    time_string=date_time_now.strftime("%H:%M")
    return weekday + ' at ' + time_string

def next_relevant_event():
    """returns the string  for next workday lunch from now"""
    weekdays=['Monday','Tuesday','Wednesday','Thursday','Friday','Monday','Monday']
    date_time=datetime.now()
    while ((date_time.weekday() in [5,6]) or is_public_holyday(date_time.date())):
        date_time+= timedelta(days=1)
    if date_time.hour <=12:
        return weekdays[date_time.weekday()] + ' lunch'
    else:
        return weekdays[date_time.weekday()+1] + ' lunch'


def return_largest_non_empty_index(features):
    result = len(features['log_num_people_11_30'].dropna())
    if features['log_num_people_11_30'].iloc[0:result].isnull().any():
        raise Exception("There are nan values in the features")
    return result

def make_prediction_next_day(features,prediction_path):
    """
    :param features:
    :param prediction_path:
    :return: num_menus_predictions, num_people_predictions (arrays of predictions
    """
    exog_feature_list = ['zuehlke_day', 'before_after_holydays']
    x=return_largest_non_empty_index(features)
    this_date=features.index[x]
    prediction_boundary=(x,x+1)
    num_people_predictions = train_sarimax_mdoel_for_prediction(features, prediction_boundary=prediction_boundary,
                                                                exog_feature_list=exog_feature_list, p=1, d=0, q=1, Ps=0, Ds=1,
                                                                Qs=1, s=5)
    with open(os.path.join('..','models','pickled_models','label_num_people_11_30_to_label_num_menus_sold.pkl'),'rb') as file:
        num_menus_estimator= pickle.load(file)
    df_predictions = pd.DataFrame(data=num_people_predictions,
                                  columns=['label_num_people_11_30'])
    num_menus_predictions= num_menus_estimator.predict(df_predictions)

    write_prediction_to_file(prediction_num_people=num_people_predictions[0], prediction_num_menus=num_menus_predictions[0],
                             integer_index=x, predictions_path=prediction_path)
    return num_menus_predictions, num_people_predictions, this_date

def write_prediction_to_file(prediction_num_people,prediction_num_menus, integer_index, predictions_path):
    predictions=pd.read_pickle(predictions_path)
    predictions.iloc[integer_index]['log_num_people_11_30']=math.log(prediction_num_people)
    predictions.iloc[integer_index]['num_people_11_30']=prediction_num_people
    predictions.iloc[integer_index]['num_menus_sold']=prediction_num_menus
    with open(predictions_path,'wb') as file:
        pickle.dump(predictions, file)






    #check_features present
    #load model: featueres -> people count 11:30
    #load model: people count -> num menus sold

def get_dummy_date():
    return date(year=2023,month=1,day=4)

def append_feature_people_count(feature_path):
    features = pd.read_pickle(feature_path)
    log_people_count=math.log(request_people_count())
    next_date=choose_next_date(features)
    features.loc[next_date, 'log_num_people_11_30']=log_people_count
    with open(feature_path, 'wb') as file:
        pickle.dump(features, file)
    return features

def request_people_count():
    #TODON
    return np.random.normal(200,100)

def choose_next_date(features):
    index_not_null=return_largest_non_empty_index(features)
    next_date=features.index[index_not_null]
    return next_date

if __name__ == '__main__':


    feature_path=os.path.join( 'data', 'features_2023.pkl')
    prediction_path=os.path.join('data','predictions_2023.pkl')
    features=append_feature_people_count(feature_path)
    num_menus_predictions, num_people_predictions,this_date= make_prediction_next_day(features, prediction_path)
    if is_public_holyday(this_date): #do another forecast
        feature_path = os.path.join('data', 'features_2023.pkl')
        prediction_path = os.path.join('data', 'predictions_2023.pkl')
        features = append_feature_people_count(feature_path)
        num_menus_predictions, num_people_predictions, this_date = make_prediction_next_day(features, prediction_path)

