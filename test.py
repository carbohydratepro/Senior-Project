import numpy as np

data = []
for i in range(2,14):
    temp=np.genfromtxt('./test.csv',delimiter=',',dtype=float,usecols=i)
    data.append(temp)

data = np.array(data)
print(type(data))