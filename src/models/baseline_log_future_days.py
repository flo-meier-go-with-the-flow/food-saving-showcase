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
import math


def eval_metrics(actual, pred):
    maloge = mean_absolute_error(actual, pred)
    rmsloge = np.sqrt(mean_squared_error(actual, pred))
    actual = np.exp(actual)
    pred = np.exp(pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2, maloge, rmsloge


def objective(params):
    # alpha=params[0]
    # l1_ratio=params[1]

    (rmse, mae, r2, maloge, rmsloge), *_ = train_model(params)
    return {'loss': rmsloge, 'status': STATUS_OK}


def train_model(params):
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        # ('scaler', StandardScaler())
    ])

    column_transformer = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features),
        ('bin', 'passthrough', bin_features)
    ])
    pipe_estimator = Pipeline([
        ('column_transformer', column_transformer),
        ('poly_features', PolynomialFeatures(poly_features_deg)),
        ('scaler', StandardScaler()),
        ('estimator', ElasticNet(**params, random_state=42))
        # ('estimator', SVR(**params))
        # ('estimator', RandomForestRegressor(**params, random_state=42))

    ])
    train_errors = []
    test_errors = []
    coefficients = []
    for i in range(0, num_validation_steps):
        pipe_estimator.fit(train_x[i], train_y[i])
        coefficients.append(pipe_estimator.named_steps['estimator'].coef_)
        # eval_metrics(train_y[i], pipe_estimator.predict(train_x[i]))
        predictions = pipe_estimator.predict(test_x[i])
        test_errors.append(eval_metrics(test_y[i], predictions))
        train_predictions = pipe_estimator.predict(train_x[i])
        train_errors.append(eval_metrics(train_y[i], train_predictions))
    rmse_train, *_ = np.mean(train_errors, axis=0)
    rmse_mean, mae_mean, r2_mean, maloge, rmsloge = np.mean(test_errors, axis=0)
    print('params:')
    print(params)
    print(f'test rmse : {rmse_mean}, train rmse : {rmse_train}')  # , mae : {mae_mean} , r2 : {r2_mean}')
    return (rmse_mean, mae_mean, r2_mean, maloge, rmsloge), pipe_estimator, coefficients


def train_final_model(params):
    column_transformer = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features),
        ('bin', 'passthrough', bin_features)
    ])
    pipe_estimator = Pipeline([
        ('column_transformer', column_transformer),
        ('poly_features', PolynomialFeatures(poly_features_deg)),
        ('scaler', StandardScaler()),
        ('estimator', ElasticNet(**params, random_state=42))
    ])
    # ('estimator', SVR(**params))
    # ('estimator', RandomForestRegressor(**params, random_state=42))
    pipe_estimator.fit(all_x, all_y)
    # eval_metrics(all_y, pipe_estimator.predict(all_x))
    coefficients = pipe_estimator.named_steps['estimator'].coef_
    return pipe_estimator, coefficients


if __name__ == '__main__':
    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)
    ##Drop NAN rows
    features_by_date.dropna(axis=0, inplace=True)
    # Drop special days
    special_days = [date(year=2022, month=11, day=25),
                    date(year=2022, month=11, day=10)]  # end of year party, zukunftstag
    features_by_date.drop(index=special_days, inplace=True)
    test_train_split = date(year=2022, month=11, day=13)
    test_data = features_by_date[features_by_date['date'] > test_train_split]
    test_week = test_data[test_data['date'] < test_train_split + timedelta(days=7)]
    train_data = features_by_date[features_by_date['date'] <= test_train_split]

    num_features = [
        'log_exp_moving_avg_0.2']  # 'exp_moving_avg_0.1','average_of_past_weeks','deviation_1_day_ago','deviation_2_day_ago','temp_deviation', 'Temperature', 'Rain Duration','Rain_half_discrete'
    cat_features = []  # 'weekday']
    cat_features_onehot = []  # 'w0','w1','w2','w3','w4']
    bin_features = ['month_quarter_0', 'month_quarter_1', 'month_quarter_2', 'month_quarter_3', 'zuehlke_day',
                    'Spring/Autumn', 'Summer',
                    'Winter']  # 'Rain_binary','Spring/Autumn','Summer','Winter','zurich_vacation', 'zuehlke_day', 'Monday','Tuesday','Wednesday','Thursday','Friday','month_quarter_0'(1,2,3)
    label = ['log_num_people_11_30']  # 'label_num_people_11_30','label_num_people_12_33','label_num_menus_sold'
    all_features = num_features + cat_features + bin_features + label
    features_for_storing_coefs = num_features + cat_features_onehot + bin_features

    train_x, test_x, train_y, test_y = [], [], [], []
    num_validation_steps = 10
    for i in range(0, num_validation_steps):
        train, test = train_test_split(train_data[all_features], random_state=i,
                                       stratify=train_data['weekday'])
        train_x.append(train.drop(label, axis=1))
        test_x.append(test.drop(label, axis=1))
        train_y.append(train[label].values.ravel())
        test_y.append(test[label].values.ravel())
    all_x = train_data[all_features].drop(label, axis=1)
    all_y = train_data[label].values.ravel()

    # hyperopt search space
    poly_features_deg = 1
    search_space_elastic_net = {
        'alpha': hp.loguniform('alpha', -2, 2),  # if I understood this correctly it returns a value beteen e^a and e^b
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0)
    }
    kernel_list = ['linear', 'rbf', 'poly']
    search_space_svr = {
        'C': hp.loguniform('C', -3, 3),
        'kernel': hp.choice('kernel', kernel_list)
    }
    num_estimator_choice = [50, 100, 200, 400]
    max_depth_choice = [2, 3, 4, 6, 8]
    search_space_random_forest = {
        'n_estimators': hp.choice('n_estimators', num_estimator_choice),
        'max_depth': hp.choice('max_depth', max_depth_choice)
    }
    best_params = fmin(
        fn=objective,
        space=search_space_elastic_net,
        algo=tpe.suggest,
        max_evals=150)
    # best_params['kernel']=kernel_list[best_params['kernel']]
    # best_params['max_depth']=max_depth_choice[best_params['max_depth']]
    # best_params['n_estimators']=num_estimator_choice[best_params['n_estimators']]
    print(f'\n\n best params: {best_params}')

    mlflow.set_experiment(
        'log_future_days:experimentation')  # 'future_days:experimentation','future_days:label_num_menus_sold_experimentation', 'elastic_net_experimentation','elastic_net_label_12_20','elastic_net_label_11_30','elastic_net_label_num_menu_sold'
    #    mlflow.set_experiment('random_forest_reg')
    # mlflow.set_experiment('SVR')

    with mlflow.start_run():
        (rmse, mae, r2, maloge, rmsloge), estimator, coefficients = train_model(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("maloge", maloge)
        mlflow.log_metric("rmsloge", rmsloge)

        mlflow.log_params(best_params)
        mlflow.log_param('poly_features_deg', poly_features_deg)
        mlflow.log_param('features and label',
                         {'num_featers': num_features, 'bin_features': bin_features, 'cat_features': cat_features,
                          'label': label})
        mlflow.sklearn.log_model(estimator, "model")
        # print(f'Intercept: {lr.intercept_} \n Coefficients: {lr.coef_}')

        # print(estimator.named_steps['estimator'].feature_importances_)
        coefs = {'coef': estimator.named_steps['estimator'].coef_,
                 'intercept': estimator.named_steps['estimator'].intercept_,
                 'all_feature_cols': features_for_storing_coefs}
        coefficients = np.array(coefficients)
        coefficients_mean = np.mean(coefficients, axis=0)
        feature_names = estimator.named_steps['poly_features'].get_feature_names_out(
            features_for_storing_coefs)
        model_coefs_mean = {key: value for key, value in zip(feature_names, coefficients_mean)}
        # with open('model_coefs.txt', 'w') as coefs_file:
        #     coefs_file.write(json.dumps(model_coefs))

        with open('model_coefs.txt', 'w') as f:
            f.write('Features: ')
            for i in all_features:
                f.write(f'{i} , ')
            f.write('\ncoef of poly2 linreg\n')
            for key, value in model_coefs_mean.items():
                f.write(f'{key} : {value}\n')
            for i, feature_name in enumerate(feature_names):
                f.write(f'{feature_name} : {coefficients_mean[i]}, individual values {coefficients[:, i]}\n')
        mlflow.log_artifact('model_coefs.txt')

        make_prediction_on_whole_dataset = True
        if make_prediction_on_whole_dataset:
            final_estimator, coefficients = train_final_model(best_params)
            x_data = features_by_date[all_features].drop(label, axis=1)
            y_data = features_by_date[label]
            y_predictions = final_estimator.predict(x_data)

            with open(os.path.join('predictions_future_days', 'x_data_' + label[0] + '.pkl'), 'wb') as file:
                pickle.dump(x_data, file)
            with open(os.path.join('predictions_future_days', 'y_data_' + label[0] + '.pkl'), 'wb') as file:
                pickle.dump(y_data, file)
            with open(os.path.join('predictions_future_days', 'y_predictions_' + label[0] + '.pkl'), 'wb') as file:
                pickle.dump(y_predictions, file)
