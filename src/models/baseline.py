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
import statsmodels


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def objective(params):
    #alpha=params[0]
    #l1_ratio=params[1]
    (rmse, mae, r2) = train_model(**params)
    return {'loss': rmse, 'status': STATUS_OK}

def train_model(alpha,l1_ratio):
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('linreg_elasticnet', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42))])
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    pipe.fit(train_x, train_y)
    predicted_qualities = pipe.predict(test_x)
    return eval_metrics(test_y, predicted_qualities)

if __name__ == '__main__':

    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)
    ##Drop NAN rows
    features_by_date.dropna(axis=0, inplace=True)


    # create datelist
    # we remove Saturday and Sundays and Public holidays by checking that at least 25 people are around
    pickle_path = os.path.join('..', '..', 'data', 'flow_counts_preprocessed')
    df = pd.read_pickle(pickle_path)
    weekdays = [0, 1, 2, 3, 4]  # mon, tue, wed, thu, fri
    date_list_all = list(df[(df['time_numeric'] == 12.33)]['date'])
    date_list_weekdays = [date for date in date_list_all if date.weekday() in weekdays]
    date_list_weekends_and_holidays = list(df[(df['time_numeric'] == 12.33) & (df['num_people_inside'] < 25)]['date'])
    date_list_holidays = [date for date in date_list_weekends_and_holidays if date.weekday() in weekdays]
    date_list_workdays = list(df[(df['time_numeric'] == 12.33) & (df['num_people_inside'] > 25)]['date'])


    # baseline with num people at 10 as only feature
    choose_features=True
    if choose_features:
        relevant_num_days_ago = [1, 2, 3, 4, 5, 6, 7, 14, 21]
        feature_list = ['label_num_people_12_33','num_people_10_00']
        feature_list.extend([str(i) + '_days_ago_12_33' for i in relevant_num_days_ago])
        feature_list.extend([str(i) + '_days_ago_norm_dif' for i in relevant_num_days_ago])
    else:
        feature_list = ['label_num_people_12_33', 'num_people_10_00']
        relevant_num_days_ago= [0]
    weekday_date_lists = {}
    for i in weekdays:
        weekday_date_lists[i] = [day for day in date_list_workdays if (day.weekday() == i) and (day >= date(year=2022,month=3,day=1)+timedelta(days=max(relevant_num_days_ago)))]

    monday_list = weekday_date_lists[1]
    data = features_by_date.loc[monday_list, feature_list]

    train, test = train_test_split(data)
    train_x = train.drop(["label_num_people_12_33"], axis=1)
    test_x = test.drop(["label_num_people_12_33"], axis=1)
    train_y = train[["label_num_people_12_33"]]
    test_y = test[["label_num_people_12_33"]]

    # hyperopt search space
    search_space = {
        'alpha':hp.loguniform('alpha', -3, 2), #if I understood this correctly it returns a value beteen e-3 and e2
        'l1_ratio':hp.uniform('l1_ratio', 0.0, 1.0)
    }
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1000)
    print(best_params)

    # print(f'Intercept: {lr.intercept_} \n Coefficients: {lr.coef_}')
