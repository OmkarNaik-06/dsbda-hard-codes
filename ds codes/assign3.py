import pandas as pd
df = pd.read_csv("C:/Users/Admin/Downloads/adult.csv")
print(df)
#summary statistics of age grouped by gender
df.groupby("gender")["age"].describe()
df.groupby("marital-status")["age"].mean()
df.groupby("marital-status")["age"].median()
#grouping can be done on multiple columns
# summary statistics of age grouped by gender & marital-status
df.groupby(["gender","marital-status"])["age"].std()
#summary statistics of age grouped by income
df.groupby("income")["age"].mean()
df.groupby(["income","gender"])["age"].mean()
df.groupby("marital-status")["marital-status"].count()
#Count number of records by category
#The value_counts() method counts the number of records for each category in a column.
df["marital-status"].value_counts()


# python program to iris dataset
import pandas as pd
d = pd.read_csv("C:/Users/Admin/Downloads/Iris.csv")
print('Iris-setosa')
setosa = d['Species'] == 'Iris-setosa'
print(d[setosa].describe())
print('\nIris-versicolor')
setosa = d['Species'] == 'Iris-versicolor'
print(d[setosa].describe())
print('\nIris-virginica')
setosa = d['Species'] == 'Iris-virginica'
print(d[setosa])
print(d[setosa].describe())

# program using groupby function
import pandas as pd
d = pd.read_csv("C:/Users/Admin/Downloads/Iris.csv")
#Species 
d.groupby(["Species"])["SepalLengthCm"].mean()
d.groupby(["Species"])["SepalLengthCm"].std()
d.groupby(["Species"])["SepalLengthCm"].describe()
d.groupby(["Species"])["SepalLengthCm"].quantile(q=0.75)
d.groupby(["Species"])["SepalLengthCm"].quantile(q=0.25)
a=d.groupby(["Species"])["SepalLengthCm"].mean()
print(a)
b=d.groupby(["Species"])["SepalLengthCm"].median()
print(b)
34
list=[a,b]
print(list)
