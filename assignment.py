import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
#%matplotlib inline
sns.set(color_codes=True)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"D:\download\Cars_data_.csv")

data.head(5)
print(data.head(5))

print(data.shape)

print(data.columns)

print(data.info())

print(data.dtypes)

print(data.count())

cols_to_drop = ['Engine Fuel Type','Number of Doors' , 'Market Category', 'Vehicle Size','Vehicle Style', 'Popularity']
data = data.drop(columns=cols_to_drop)
print(data.head())

rename_cols = {    "Engine HP": "HP",
    "Engine Cylinders": "Cylinders",
    "Transmission Type": "Transmission",
    "Driven_Wheels": "Drive Mode",
    "highway MPG": "Highway MPG",
    "city mpg": "City MPG",
    "MSRP": "Price"}
data = data.rename(columns=rename_cols)
print(data.head())

print(data.shape[0])
data = data.drop_duplicates()
print(data.head())
print(data.shape[0])

print(data.isnull().sum())
data = data = data.dropna()
print(data.isnull().sum())

print(data.shape)
print(data.describe())

plt.figure(figsize=(10,5))
sns.boxplot(x=data["Price"])
plt.show()



plt.figure(figsize=(10, 5))
sns.boxplot(x=data["HP"])
plt.show()

numeric_columns = data.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumeric columns:")
print(numeric_columns)

l = list(numeric_columns)

print("\nList of numeric columns:")
print(l)


# define Q1 and Q2
Q1 = data[l].quantile(0.25)
Q3 = data[l].quantile(0.75)

# define IQR (interquantile range)
IQR = Q3 - Q1

print("\nQ1:")
print(Q1)

print("\nQ3:")
print(Q3)

print("\nIQR:")
print(IQR)

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower limits:")
print(lower_limit)
print("\nUpper limits:")
print(upper_limit)

# define df2 after removing outliers
data2 = data[
    ~( (data[l] < lower_limit) |
     (data[l] > upper_limit)
  ).any(axis=1)
]

print("\nShape of original dataframe:")
print(data.shape)

print("\nShape after removing outliers:")
print(data2.shape)

for i in data2.columns:
    print ("--------------- %s ----------------" % i)
    print(data2[i].value_counts())
    print()

    plt.figure(figsize=(8,5))
sns.histplot(data2["HP"], kde=True)

plt.title("Distribution of HP")
plt.xlabel("HP")
plt.show()



c=1
plt.figure(figsize=(15,10))
for i in l:
    plt.subplot(2,3,c)
    sns.histplot(data2[i], kde=True)
    plt.title(i)
    c += 1

plt.tight_layout()
plt.show()


plt.figure(figsize=(15,5))

sns.countplot( data=data, x="Transmission", hue="Drive Mode" )
plt.title("Transmission Type by Drive Mode")
plt.xlabel("Transmission")
plt.ylabel("Count")
plt.show()


fig, ax = plt.subplots(figsize=(10,6))

sns.scatterplot( data=data, x="HP", y="Price", ax=ax )
plt.title("Horsepower vs Price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()



plt.figure(figsize=(10, 6))
sns.barplot( data=data, x="Cylinders", y="Price" )
plt.title("Average Price by Number of Cylinders")
plt.xlabel("Number of Cylinders")
plt.ylabel("Average Price")
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot( data=data, y="Transmission" )
plt.title("Number of Cars by Transmission Type")
plt.xlabel("Number of Cars")
plt.ylabel("Transmission")
plt.show()




corr = data[l].corr()

# print corr
print("\nCorrelation matrix:")
print(corr)


plt.figure(figsize=(12, 8))
sns.heatmap( corr, cmap="BrBG", annot=True )
plt.title("Correlation Heatmap")
plt.show()
