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
from statsmodels.tsa.seasonal import STL


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    maloge = np.mean(np.abs(np.log(pred / actual)))
    return rmse, mae, r2, maloge


def objective(params):
    # alpha=params[0]
    # l1_ratio=params[1]

    (rmse, mae, r2, maloge), *_ = train_model(**params)
    return {'loss': rmse, 'status': STATUS_OK}


def train_model(alpha=1, l1_ratio=0.5, exp_rolling_avg_alpha=0.2, seasonal=13):
    predictions = []
    coefficients = []

    test_set = range(train_test_boundary, len(features_by_date))
    for i in test_set:
        stl = STL(features_by_date.iloc[:i]['filled_num_people_11_30'], period=5, seasonal=seasonal, robust=True)
        stl_res = stl.fit()
        stl_res.seasonal[features_by_date.iloc[i]['date']] = stl_res.seasonal[
            features_by_date.iloc[i]['date'] - timedelta(days=7)]
        labels = features_by_date.iloc[:i + 1]['filled_num_people_11_30'] - stl_res.seasonal
        features_by_date['labels'] = labels
        features_by_date['rolling_avg'] = labels.rolling(window=rolling_avg).mean().shift(1)
        features_by_date['exp_rolling_avg'] = labels.ewm(alpha=exp_rolling_avg_alpha).mean().shift(1)
        features_by_date['dif_labels_rolling_avg'] = features_by_date['labels'] - features_by_date['rolling_avg']
        for j in range(1, 6):
            features_by_date[f'dif_labels_rolling_avg_{j}_days_ago'] = features_by_date['dif_labels_rolling_avg'].shift(j)

        column_transformer = ColumnTransformer([
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(), cat_features),
            ('bin', 'passthrough', bin_features)
        ])
        pipe_estimator = Pipeline([
            ('column_transformer', column_transformer),
            ('poly_features', PolynomialFeatures(poly_features_deg)),
            ('scaler', StandardScaler()),
            ('estimator', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42))
            # ('estimator', SVR(**params))
            # ('estimator', RandomForestRegressor(**params, random_state=42))

        ])
        pipe_estimator.fit(features_by_date[10:i], labels[10:i])
        coefficients.append(pipe_estimator.named_steps['estimator'].coef_)
        if i == test_set[-1]:
            train_predictions = pipe_estimator.predict(features_by_date[10:i]) + stl_res.seasonal[10:i]
            print(f'Train_error: {eval_metrics(features_by_date[10:i]["filled_num_people_11_30"], train_predictions)}')
        predictions.append(pipe_estimator.predict(features_by_date.iloc[i:i + 1])[0] + stl_res.seasonal.iloc[-1])
    errors = eval_metrics(features_by_date.iloc[test_set]['filled_num_people_11_30'], np.array(predictions))
    print(f'Test_error: {errors}')
    print(f'params:  alpha {alpha}, l1_ratio {l1_ratio}, exp_alpha {exp_rolling_avg_alpha}, seasonal {seasonal}')
    return errors, pipe_estimator, coefficients


if __name__ == '__main__':
    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)
    df_num_people = pd.Series(data=features_by_date['filled_num_people_11_30'].values, index=features_by_date.index)

    # params:

    train_test_boundary = 150
    rolling_avg = 5
    poly_features_deg = 2

    num_features = [
        'rolling_avg', 'dif_labels_rolling_avg_1_days_ago']  # 'exp_rolling_avg','exp_moving_avg_0.1','average_of_past_weeks','deviation_1_day_ago','deviation_2_day_ago','temp_deviation', 'Temperature', 'Rain Duration','Rain_half_discrete'
    cat_features = []  # 'weekday']
    cat_features_onehot = []  # 'w0','w1','w2','w3','w4']
    bin_features = [
        'zuehlke_day']  # 'month_quarter_0','month_quarter_1','month_quarter_2','month_quarter_3','zuehlke_day','Spring/Autumn','Summer','Winter']  # 'Rain_binary','Spring/Autumn','Summer','Winter','zurich_vacation', 'zuehlke_day', 'Monday','Tuesday','Wednesday','Thursday','Friday','month_quarter_0'(1,2,3)
    label = ['labels']
    all_features = num_features + cat_features + bin_features + label
    features_for_storing_coefs = num_features + cat_features_onehot + bin_features

    # hyperopt search space
    seasonal_param_list=[7, 13, 23]
    search_space_elastic_net = {

        'alpha': hp.loguniform('alpha', -2, 2),  # if I understood this correctly it returns a value beteen e^a and e^b
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
        #'exp_rolling_avg_alpha': hp.uniform('exp_rolling_avg_alpha', 0.0, 1.0),
        'seasonal': hp.choice('seasonal',seasonal_param_list )
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
        max_evals=100)
    # best_params['kernel']=kernel_list[best_params['kernel']]
    # best_params['max_depth']=max_depth_choice[best_params['max_depth']]
    # best_params['n_estimators']=num_estimator_choice[best_params['n_estimators']]
    best_params['seasonal']=seasonal_param_list[best_params['seasonal']]
    print(f'\n\n best params: {best_params}')

    mlflow.set_experiment(
        'future_days:stl_experimentation')  # 'future_days:experimentation','future_days:label_num_menus_sold_experimentation', 'elastic_net_experimentation','elastic_net_label_12_20','elastic_net_label_11_30','elastic_net_label_num_menu_sold'
    #    mlflow.set_experiment('random_forest_reg')
    # mlflow.set_experiment('SVR')

    with mlflow.start_run():
        print(best_params)
        (rmse, mae, r2, maloge), estimator, coefficients = train_model(**best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("maloge", maloge)

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
