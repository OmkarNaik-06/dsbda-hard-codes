import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = ""
df = pd.read_csv(path)
print(df)

df.is_null()
print(df)

df.is_null().sum().sum()

df.describe()

df.describe(include=['object'])

df.dtypes()

df.shape()
df.size()

