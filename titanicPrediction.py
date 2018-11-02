# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:14:02 2018

@author: Rachid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Datas

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# Missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = round(percent_1,1).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)

# comments : with 77.1% of missing values, 'Cabin' columns need to be drop from the dataset

# What features could contribute to high rate of survival
# we don't need PassengerId , Name and ticker number

# Age and Sex
survived = 'survived'
not_survived = 'not survived'

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(),bins = 18,
                 label = 'Survived', ax = axes[0], kde=False)

ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(),bins = 40,
                 label = 'Not Survived', ax = axes[0], kde=False)

ax.legend()
ax.set_title('Female')

ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(),bins = 18,
                 label = 'Survived', ax = axes[1], kde=False)

ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(),bins = 40,
                 label = 'Not Survived', ax = axes[1], kde=False)

ax.legend()
_ = ax.set_title('Male')

plt.show()

# Embarked Pclass and Sex
FacetGrid = sns.FacetGrid(train_df, row = 'Embarked',size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived','Sex', palette=None,
              order=None, hue_order=None)
FacetGrid.add_legend()

# PClass
sns.barplot(x='Pclass',y='Survived', data=train_df)

# Pclass / Survived / Age
grid = sns.FacetGrid(train_df, col='Survived', row = 'Pclass', size = 2.2, aspect = 1.6)
grid.map(plt.hist, 'Age', alpha = .5, bins = 20)
grid.add_legend()

# sibsp : of siblings / spouses aboard the Titanic
# parch : of parents / children aboard the Titanic

data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
    
train_df['not_alone'].value_counts()

axes = sns.factorplot('relatives', 'Survived', data = train_df, aspect = 2.5)

# data preprocessing
train_df = train_df.drop(['PassengerId'], axis=1)

# fix Cabin issue
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)
    
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)

# Fix Age issue
data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    
    rand_age = np.random.randint(mean-std, mean+std, size = is_null)
    
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

# test
train_df["Age"].isnull().sum()























































