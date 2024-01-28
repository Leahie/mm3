import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools 
from datetime import timedelta 
plt.style.use("seaborn")
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

plt.plot(times_X, pop_Y)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Global Population Since 1950')
plt.show()

