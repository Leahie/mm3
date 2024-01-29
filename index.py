import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools 
from datetime import timedelta 
from sklearn.linear_model import LinearRegression 
FS = (16, 9)


df = pd.read_csv('./world_population_by_year_1950_2023.csv')

df2 = df.loc[:, '1950': '2023':1]

sums = df2.sum().rename('total')
s = sums.to_frame()
"""
times_X = list(s.index)
times_X = np.array(times_X, dtype=int)
print(times_X)
pop_Y = s['total'].tolist()

plt.plot(times_X, pop_Y)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Global Population Since 1950')
plt.show() """

def plot_ratios(data, pop_min = 0, pop_max = None):
    # proportional growth rate with respect to D 
    data = data[data>pop_min]
    if pop_max is not None: data = data[data<pop_max]
    slopes = 0.5 *(data.diff(1)-data.diff(-1))
    print(data.diff(1)- data.diff(-1))
    ratios = slopes/ data
    x = data.values[1:-1] # returns array of data 
    y = ratios.values[1:-1] # returns array from ratios
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    plt.plot(x, y, 'o')
    ax.set_xlabel('D(t)')
    ax.set_ylabel('Ratios of slopes to function values')
    return x, y

def linear_regression(x,y):
    # returning the coefficients a and b
    X = x.reshape(-1,1)
    reg = LinearRegression(fit_intercept=True, normalize=True)
    reg.fit(X,y)
    a = reg.coef_[0]
    b = reg.intercept_
    y_hat = a*x + b
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1) 
    plt.plot(x,y, 'o')
    plt.plot(x, y_hat, '-', label='Linear Reg')
    ax.set_xlabel('D(t)')
    ax.set_ylabel('Ratios of slopes to function values')
    ax.legend()
    return a, b

def t0(a, b, data, pop_min=0, pop_max=None, t0=0):
    # find the best value based on the Lin_Reg
    k =  b
    L = -b/a
    data = data[data>pop_min]
    if pop_max is not None: data = data[data<pop_max]
    logfunc = L/ (1.0 + np.exp(-k*(np.arange(len(data))-t0)))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1) 
    plt.plot(logfunc, 'o')
    plt.plot(data.values, 'd')
    return L, k

def extended_plot(data, pop_min, L, k, t0, years_before = 74, years_after = 30, figsize = (16, 9) ):
    # plot log function against original data 
    data_org = data.copy(deep = True)
    data = data[data>pop_min]
    data = data.to_frame('Total')
    data_start, data_end = data.index.min(), data.index.max()
    start = data_start - years_before 
    end = data_end + years_after
    ix = pd.date_range(start=start, end=end, freq = 'D')
    data = data.reindex(ix)
    data['idx'] = np.arange(len(data))
    data['idx'] -= data.loc[data_start, 'idx'] # normalizing 
    data['logistic'] = L/ (1.0 + np.exp(-k*(np.arange(len(data))-t0)))
    ax =    






data = s['total']
x,y = plot_ratios(data)
print(x)

print(y)