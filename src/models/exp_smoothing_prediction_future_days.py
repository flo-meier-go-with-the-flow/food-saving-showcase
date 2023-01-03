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
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    maloge = np.mean(np.abs(np.log(pred / actual)))
    return rmse, mae, r2, maloge


def objective(params):
    # alpha=params[0]
    # l1_ratio=params[1]

    (rmse, mae, r2, maloge) = train_exp_smoothing_model(features_by_date=features_by_date,
                                                                 train_test_boundary=train_test_boundary)
    return {'loss': rmse, 'status': STATUS_OK}


def train_exp_smoothing_model(features_by_date, train_test_boundary=150):
    predictions = []
    coefficients = []

    test_set = range(train_test_boundary, len(features_by_date))
    for i in test_set:
        exp_smoothing_model = ExponentialSmoothing(features_by_date.iloc[:i][label], trend='add',
                                                   seasonal_periods=5, seasonal=seasonal_add_or_mul).fit()

        predictions.append(exp_smoothing_model.predict(start=i, end=i+1)[0])

        # compute train error
        if i == test_set[-1]:
            train_predictions = exp_smoothing_model.predict(start=test_set[0], end=i - 1)
            labels = features_by_date.iloc[test_set[:-1]][label]
            print(f'Train_error: {eval_metrics(labels, train_predictions)}')
    errors = eval_metrics(features_by_date.iloc[test_set][label], np.array(predictions))

    print(f'Test_error: {errors}')

    return errors


if __name__ == '__main__':
    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)

    train_test_boundary = 150
    seasonal_add_or_mul='add'
    exog_feature_list = ['zuehlke_day']
    label='filled_num_people_11_30'

    sarimax_parameter_list = [0]
    search_space_sarimax = {

        # 'exp_rolling_avg_alpha': hp.uniform('exp_rolling_avg_alpha', 0.0, 1.0),
        'p': hp.choice('p', sarimax_parameter_list),
        'd': hp.choice('d', sarimax_parameter_list),
        'q': hp.choice('q', sarimax_parameter_list),
        'Ps': hp.choice('Ps', sarimax_parameter_list),
        'Ds': hp.choice('Ds', sarimax_parameter_list),
        'Qs': hp.choice('Qs', sarimax_parameter_list)
    }
    best_params = fmin(
        fn=objective,
        space=search_space_sarimax,
        algo=tpe.suggest,
        max_evals=1)
    print(f'\n\n best params: {best_params}')

    mlflow.set_experiment(
        'future_days:exp_smoothing_experimentation')  # 'future_days:experimentation','future_days:label_num_menus_sold_experimentation', 'elastic_net_experimentation','elastic_net_label_12_20','elastic_net_label_11_30','elastic_net_label_num_menu_sold'
    #    mlflow.set_experiment('random_forest_reg')
    # mlflow.set_experiment('SVR')

    with mlflow.start_run():
        print(best_params)
        (rmse, mae, r2, maloge) = train_exp_smoothing_model(features_by_date,
                                                                                  train_test_boundary=train_test_boundary)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("maloge", maloge)

        #mlflow.log_params(best_params)
        mlflow.log_param('params',
                         {'seasonal add or mul': seasonal_add_or_mul})

        #with open('model_summary.txt', 'w') as f:
        #    f.write(res.summary().as_text())
        #mlflow.log_artifact('model_summary.txt')
