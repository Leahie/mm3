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

times_X = list(s.index)
times_X = np.array(times_X, dtype=int)
print(times_X)
pop_Y = s['total'].tolist()
"""plt.plot(times_X, pop_Y)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Global Population Since 1950')
plt.show() """

def normalize(arr, t_min, t):
    norm_arr = []
    diff = t - t_min
    diff_arr =(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def plot_ratios(data, pop_min = 0, pop = None):
    # proportional growth rate with respect to D 
    data = data[data>pop_min]
    if pop is not None: data = data[data<pop]
    slopes = 0.5 *(data.diff(1)-data.diff(-1))
    ratios = slopes/ data
    x = data.values[1:-1] # returns array of data 
    y = ratios.values[1:-1] # returns array from ratios
    return x, y
""" fig = plt.figure(figsize=(8,8)) = fig.add_subplot(1,1,1)
    plt.plot(x, y, 'o').set_xlabel('D(t)').set_ylabel('Ratios of slopes to function values') """
    

def linear_regression(x,y):
    # returning the coefficients a and b
    X = x.reshape(-1,1)
    reg = LinearRegression(fit_intercept=True)
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
"""      """

def t(a, b, data, pop_min=0, pop=None, t0=0):
    # find the best value based on the Lin_Reg
    k =  b
    L = -b/a
    data = data[data>pop_min]
    if pop is not None: data = data[data<pop]
    logfunc = L/ (1.0 + np.exp(-k*(np.arange(len(data))-t0)))  
    return L, k
""" fig = plt.figure(figsize=(8,8)) = fig.add_subplot(1,1,1) 
    plt.plot(logfunc, 'o', label = "logfunc")
    plt.plot(data.values, 'd').legend()     """


def extended_plot(data, L, k, t0, pop_min =0, years_before = 100, years_after = 200, figsize = (16, 9) ):
    # plot log function against original data 
    data_org = data.copy(deep = True)
    data = data[data>pop_min]
    data = data.to_frame('Total')
    data_start, data_end = data.index.min(), data.index.max()

    start = int(data_start) - int(years_before)
    end = int(data_end) + int(years_after )

    data = data.reindex(range(start, end))
    data['idx'] = np.arange(len(data))
    data['idx'] -= data.loc[int(data_start), 'idx'] # normalizing 
    data['logistic'] = L/ (1.0 + np.exp(-k*(data['idx'].values-t0)))
    ax = data['logistic'].plot(figsize=figsize, logy=False, label="log function")

    data_org['idx'] = np.arange(1950, 2024)
    print(times_X, pop_Y)
    _ = plt.plot(times_X, pop_Y, "d", alpha=0.5, label = "Original Data")
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Predicted Global Population From 1850-2223')

""" 
data = s['total']
x,y = plot_ratios(data)
a,b = linear_regression(x,y)
t0 = 60
L,k = t(a,b, data, t0=t0 )
print(L, k)
#extended_plot(data, L=L, k=k, t0=t0, figsize=FS)
plt.show()
 """

df = pd.read_csv('./world_population_by_year_1950_2023.csv')
dfMalaria = pd.read_csv('./incedenceOfMalaria.csv')
dfTuber = pd.read_csv('./incedenceOfTuberculosis.csv')
dfDoctors = pd.read_csv('./medicalDoctors.csv')

print(dfMalaria)
print(dfTuber)
print(dfDoctors)
print(df)

# process function
def process(data, pop):
    data_start, data_end = data['Period'].min(), data['Period'].max()
    population = []
    value = []
    print(data_start, data_end)
    for i in range(data_start, data_end+1): # i is the value of the year 
        col = data.loc[data['Period'] == i]
        # Getting Population
        values = pop.loc[pop['country'].isin(col['Location'].to_numpy())]
        total_pop = sum(values[str(i)])
        population.append(total_pop)
        # Get Tooltip Value 
        Tooltip = col['First Tooltip'].apply(lambda a: float((a.split())[0]))
        col['Tooltip'] = Tooltip
        value.append(sum(col['Tooltip']))
    return population, value


print(process(dfTuber, df))