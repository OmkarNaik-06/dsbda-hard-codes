import pandas as pd
df = pd.read_csv("")

print(df)

q1 = df.mathscore.quantile(0.25)
q3 = df.mathscore.quantile(0.75)
print(q1,q3)

IQR = q3-q1
print(IQR)

lower_limit = q1 - 1.5*IQR
upper_limit = q3 + 1.5*IQR
print(lower_limit,upper_limit)
 
df[(df.mathscore<lower_limit)|(df.mathscore>upper_limit>)]

df[(df.mathscore>lower_limit)&(df.mathscore<upper_limit>)]

df.describe()

df['zscore']=(df.mathscore - df.mathscore.mean())/df.mathscore.std()

import seaborn as sns
sns.boxplot(df['mathscore'])
