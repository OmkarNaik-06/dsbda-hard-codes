import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataset = sns.load_dataset('titanic')
dataset.head()

sns.barplot(x='sex',y='age',data=dataset)

sns.catplot(x='sex',hue="survived",kind='count',data=dataset)

sns.histplot(data=dataset,x='fare')

sns.histplot(data=dataset,x='fare',binwidth=30)




