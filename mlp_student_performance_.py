import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import classification_report

# load datasets for two subjects, Math and Portuguese
mat = pd.read_csv("dataset/student_performance/student-mat.csv", sep=',')
por = pd.read_csv("dataset/student_performance/student-por.csv", sep=',')

mat.describe()

# merge datasets
df = pd.concat([mat,por])

df.describe()
