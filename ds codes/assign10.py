import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data1 = pd.read_csv("")
data1.head()

print(data1.columns())

data1.info()

data1.dtypes()

data1.hist()

fig,axes = plt.subplots(2,2,figsize=(16,8))
axes[0,0].set_tile("distribution of first column")
axes[0,0].hist(data1["sepalLengthCm"])
axes[0,1].set_tile("distribution of second column")
axes[0,1].hist(data1["sepalWidthCm"])
axes[1,0].set_tile("distribution of third column")
axes[1,0].hist(data1["petalLengthCm"])
axes[1,1].set_tile("distribution of fourth column")
axes[1,1].hist(data1["petalWidthCm"])

data_to_plot = [data1["sepalLengthCm"], data1["sepalWidthCm"],data1["petalLengthCm"],data1["petalWidthCm"]]
fig = plt.figure(1,figsize=(12,8))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_to_plot)

sns.boxplot(data1['sepalWidthCm'])

print(np.where(data1['sepalWidthCm']>4.0))