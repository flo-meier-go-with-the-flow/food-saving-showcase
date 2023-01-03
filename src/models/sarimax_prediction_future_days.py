import pickle

import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima.arima import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from datetime import date
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

import os


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    maloge = np.mean(np.abs(np.log(pred / actual)))
    return rmse, mae, r2, maloge


def objective(params):
    # alpha=params[0]
    # l1_ratio=params[1]

    (rmse, mae, r2, maloge), aic, *_ = train_sarimax_model(features_by_date=features_by_date,
                                                           train_test_boundary=train_test_boundary,
                                                           exog_feature_list=exog_feature_list, **params)
    return {'loss': rmse, 'status': STATUS_OK}


def train_sarimax_model(features_by_date, p=1, d=0, q=1, Ps=0, Ds=1, Qs=1, s=5, train_test_boundary=150,
                        exog_feature_list=[]):
    order = (p, d, q)
    seasonal_order = (Ps, Ds, Qs, s)
    print(f'params: {order}, {seasonal_order}')

    log_predictions = []
    coefficients = []

    test_set = range(train_test_boundary, len(features_by_date))
    for i in test_set:
        df = features_by_date['log_num_people_11_30']
        exog = features_by_date[exog_feature_list]

        sarimax_model = ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
        res = sarimax_model.fit(y=df.iloc[:i], X=exog.iloc[:i])
        log_predictions.append(res.predict(n_periods=1, X=exog[i:i + 1])[0])

        # compute train error
        if i == test_set[-1]:
            train_predictions = res.predict_in_sample(X=exog, start=test_set[0], end=i - 1)
            labels = features_by_date.iloc[test_set[:-1]]["log_num_people_11_30"]
            print(f'Train_log_error: {eval_metrics(labels, train_predictions)}')
            print(f'Train_error: {eval_metrics(np.exp(labels), np.exp(train_predictions))}')
    log_errors = eval_metrics(features_by_date.iloc[test_set]['log_num_people_11_30'], np.array(log_predictions))
    predictions = np.exp(np.array(log_predictions))
    errors = eval_metrics(features_by_date.iloc[test_set]['filled_num_people_11_30'], predictions)

    print(f'Test_log_error: {log_errors}')
    print(f'Test_error: {errors}')
    print(f'AIC: {res.aic()}')

    return errors, res.aic(), log_errors, res, predictions


def num_menus_sold_error(predictions, train_test_boundary, features_df):
    with open(os.path.join('pickled_models', 'label_num_people_11_30_to_label_num_menus_sold.pkl'),
              'rb') as file:  # num_features[0] + '_to_' + label[0] + '
        num_menus_estimator = pickle.load(file)
    num_features = ['label_num_people_11_30']  # ,'temp_deviation', 'Temperature', 'Rain Duration']
    cat_features = []  # 'weekday']
    bin_features = []
    df_predictions = pd.DataFrame(index=features_df.iloc[train_test_boundary:].index, data=predictions,
                                  columns=['label_num_people_11_30']) # to bring the predicted for 11:30 into tbe same format as the estimator was trained on
    menus_predictions = num_menus_estimator.predict(df_predictions)
    df_predictions = pd.DataFrame(index=features_df.iloc[train_test_boundary:].index, data=menus_predictions,
                                  columns=['predictions'])
    df = features_df.iloc[train_test_boundary:]['label_num_menus_sold'].copy()
    df_predictions = df_predictions.join(df)
    df3 = df_predictions.dropna(axis=0)
    return eval_metrics(df3['predictions'], df3['label_num_menus_sold']), df_predictions


if __name__ == '__main__':
    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)

    train_test_boundary = 100
    exog_feature_list = [
        'zuehlke_day']  # 'zuehlke_day''Rain_binary','Spring/Autumn','Summer','Winter','zurich_vacation', 'zuehlke_day', 'Monday','Tuesday','Wednesday','Thursday','Friday','month_quarter_0'(1,2,3)
    hyperparameter_opt = False
    if hyperparameter_opt:

        sarimax_parameter_list = [0, 1]
        sarimax_parameter_list2 = [0, 1, 2]

        search_space_sarimax = {
            # 'exp_rolling_avg_alpha': hp.uniform('exp_rolling_avg_alpha', 0.0, 1.0),
            'p': hp.choice('p', sarimax_parameter_list2),
            'd': hp.choice('d', sarimax_parameter_list),
            'q': hp.choice('q', sarimax_parameter_list2),
            'Ps': hp.choice('Ps', sarimax_parameter_list2),
            'Ds': hp.choice('Ds', sarimax_parameter_list),
            'Qs': hp.choice('Qs', sarimax_parameter_list2)
        }
        best_params = fmin(
            fn=objective,
            space=search_space_sarimax,
            algo=tpe.suggest,
            max_evals=100)
        print(f'\n\n best params: {best_params}')
    else:
        best_params = {'p': 0, 'd': 0, 'q': 1, 'Ps': 0, 'Ds': 1, 'Qs': 1}

    mlflow.set_experiment(
        'future_days:sarimax_experimentation')  # 'future_days:experimentation','future_days:label_num_menus_sold_experimentation', 'elastic_net_experimentation','elastic_net_label_12_20','elastic_net_label_11_30','elastic_net_label_num_menu_sold'
    #    mlflow.set_experiment('random_forest_reg')
    # mlflow.set_experiment('SVR')

    with mlflow.start_run():
        print(best_params)
        (rmse, mae, r2, maloge), aic, log_errors, res, predictions = train_sarimax_model(features_by_date, s=5,
                                                                                         train_test_boundary=train_test_boundary,
                                                                                         exog_feature_list=exog_feature_list,
                                                                                         **best_params)

        (menus_rmse, menus_mae, menus_r2, menus_maloge),menus_predictions = num_menus_sold_error(predictions, train_test_boundary,
                                                                             features_by_date)

        mlflow.log_metric(" menus_rmse", menus_rmse)
        mlflow.log_metric(" menus_mae", menus_mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("maloge", maloge)
        mlflow.log_metric('aic', aic)

        mlflow.log_params(best_params)
        mlflow.log_param('exog_features',
                         {'exog_features': exog_feature_list})

        with open('model_summary.txt', 'w') as f:
            f.write(res.summary().as_text())
        mlflow.log_artifact('model_summary.txt')

        with open(os.path.join('predictions_future_days', 'sarimax_num_menus.pkl'), 'wb') as file:  # num_features[0] + '_to_' + label[0] + '
            pickle.dump(menus_predictions, file)