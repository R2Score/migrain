# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:02:30 2022

@author: Roman Semenov
"""

'''Прогнозирование показателей миграции в РФ с использованием моделей машинного обучения'''


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

def lin_reg (x_train, x_test, y_train, y_test, y_val): 
    reg = LinearRegression() 
    reg.fit(x_train, y_train) 
    y_pred = reg.predict(x_train) 
    y_pred2 = reg.predict(x_test) 
    y_pred3 = reg.predict(x_val) 
    print('===LinearRegression===')
    print('Коэффициент b - сдвиг прямой из уровнения y = m*x + b:', reg.intercept_)
    print('Коэффициент m  - наклон прямой из уровнения y = m*x + b:', reg.coef_) 
    print('----------------------')
    print('Оценка качества обучения модели R^2:', r2_score (y_train, y_pred))
    print('Точность модели MAPE (отклонение от факта):', mean_absolute_percentage_error(y_test, y_pred2))
    print('Прогнозирование MIGRAIN на заданный период:', y_pred3)
    print()
    plt.plot(x_train, y_train, 'o')
    plt.plot(x_train, y_pred)
    plt.xlabel('VVP')
    plt.ylabel('MIGRAIN')
    plt.show()
    return y_pred


def neighbor (x_train, x_test, y_train, y_test, n, y_val):
    reg = KNeighborsRegressor(n_neighbors=n)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_train)
    y_pred2 = reg.predict(x_test)
    y_pred3 = reg.predict(x_val) 
    print('===KNeighborsRegressor===')
    print('Оценка качества обучения модели R^2:', r2_score(y_train, y_pred))
    print('Точность модели MAPE (отклонение от факта):', mean_absolute_percentage_error(y_test, y_pred2))
    print('Прогнозирование MIGRAIN на заданный период:', y_pred3)
    print()
    return y_pred


def des_tree (x_train, x_test, y_train, y_test, y_val):
    reg = DecisionTreeRegressor(min_samples_split = 3, min_samples_leaf = 3)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_train)
    y_pred2 = reg.predict(x_test)
    y_pred3 = reg.predict(x_val)
    print('===DecisionTreeRegressor===')
    print('Оценка качества обучения модели R^2:', r2_score(y_train, y_pred))
    print('Точность модели MAPE (отклонение от факта):', mean_absolute_percentage_error(y_test, y_pred2))
    print('Прогнозирование MIGRAIN на заданный период:', y_pred3)
    print()
    return y_pred

def RFR (x_train, x_test, y_train, y_test, y_val):
    reg = RandomForestRegressor(min_samples_split = 3, min_samples_leaf = 3, n_estimators = 16)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_train)
    y_pred2 = reg.predict(x_test)
    y_pred3 = reg.predict(x_val)
    print('===RandomForestRegressor===')
    print('Оценка качества обучения модели R^2:', r2_score(y_train, y_pred))
    print('Точность модели MAPE (отклонение от факта):', mean_absolute_percentage_error(y_test, y_pred2))
    print('Прогнозирование MIGRAIN на заданный период:', y_pred3)
    print()
    return y_pred

df = pd.read_csv('socio-economic_data.csv', sep = ';', encoding = 'cp1251')
df_stat = df[['MIGRAIN', 'KR', 'M2', 'FW', 'REZ', 'VVP']]
print(df_stat.describe())
print(df_stat.info())

df = df[['MIGRAIN', 'KR', 'M2', 'FW', 'REZ', 'VVP']]

X = df[['VVP']]
y = df['MIGRAIN']

x_train = X.iloc()[:28] 
x_test = X.iloc()[28:32] 

y_train = y.iloc()[:28]
y_test = y.iloc()[28:32]

x_val = X.iloc()[32:]

y_liner = lin_reg(x_train, x_test, y_train, y_test, x_val)

X = df[['VVP', 'KR', 'M2', 'FW', 'REZ']]
y = df['MIGRAIN'] 

x_train = X.iloc()[:28]
x_test = X.iloc()[28:32]

y_train = y.iloc()[:28]
y_test = y.iloc()[28:32]

x_val = X.iloc()[32:]

y_neighbor = neighbor (x_train, x_test, y_train, y_test, 5, x_val)
y_des_tree = des_tree (x_train, x_test, y_train, y_test, x_val)
y_RR = RFR (x_train, x_test, y_train, y_test, x_val)

sns.heatmap(df.corr(), annot = True)