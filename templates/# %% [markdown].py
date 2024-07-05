# %% [markdown]
# # <center>Indian Liver Patient Records EDA</center>

# %% [markdown]
# # Dataset Description
# This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.
# 
# Any patient whose age exceeded 89 is listed as being of age "90".
# 
# **Columns:**
# 
# * Age of the patient
# * Gender of the patient
# * Total Bilirubin
# * Direct Bilirubin
# * Alkaline Phosphotase
# * Alamine Aminotransferase
# * Aspartate Aminotransferase
# * Total Protiens
# * Albumin
# * Albumin and Globulin Ratio
# * Dataset: field used to split the data into two sets (patient with liver disease, or no disease)
# 
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
data =pd.read_csv('./data.csv')

# %%
data.head()

# %%
data.shape

# %%
data.info()

# %%
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data.head()

# %%
data.describe().transpose()

# %%
data.isnull().sum()

# %%
data.duplicated().sum()

# %%
data.drop_duplicates()

# %% [markdown]
# # Target Variable

# %%
data.Dataset.value_counts()

# %%
data.shape

# %% [markdown]
# # Age Variable

# %%
data['Age'].describe()

# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Age', data = data,color='maroon')
plt.title('Boxplot for Age Variable')
plt.show()

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Age'], kde = True,color='maroon')
plt.title('Histogram for Age Variable')
plt.show()


# %% [markdown]
# # Gender

# %%
plt.figure(figsize=[12,4])
sns.countplot(y = data['Gender'],palette='Greens')
plt.title('Histogram for Age Variable')
plt.show()


# %% [markdown]
# # Total_Bilirubin

# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Total_Bilirubin', data = data,color='maroon')
plt.title('Boxplot for Total_Bilirubin Variable')
plt.show()

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Total_Bilirubin'], kde = True,color='maroon')
plt.title('Histogram for Total_Bilirubin Variable')
plt.show()


# %% [markdown]
# # Direct_Bilirubin

# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Direct_Bilirubin', data = data,color='maroon')
plt.title('Boxplot for Direct_Bilirubin Variable')
plt.show()

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Direct_Bilirubin'], kde = True,color='maroon')
plt.title('Histogram for Direct_Bilirubin Variable')
plt.show()


# %% [markdown]
# # Alkaline_Phosphotase

# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Alkaline_Phosphotase', data = data,color='maroon')
plt.title('Boxplot for Alkaline_Phosphotase Variable')
plt.show()

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Alkaline_Phosphotase'], kde = True,color='maroon')
plt.title('Histogram for Alkaline_Phosphotase Variable')
plt.show()


# %% [markdown]
# # Alamine_Aminotransferase
# 

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Alamine_Aminotransferase'], kde = True,color='maroon')
plt.title('Histogram for Alamine_Aminotransferase Variable')
plt.show()


# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Alamine_Aminotransferase', data = data,color='maroon')
plt.title('Boxplot for Alamine_Aminotransferase Variable')
plt.show()

# %% [markdown]
# # Aspartate_Aminotransferase

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Aspartate_Aminotransferase'], kde = True,color='maroon')
plt.title('Histogram for Aspartate_Aminotransferase Variable')
plt.show()


# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Aspartate_Aminotransferase', data = data,color='maroon')
plt.title('Boxplot for Aspartate_Aminotransferase Variable')
plt.show()

# %% [markdown]
# # Total_Protiens

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Total_Protiens'], kde = True,color='maroon')
plt.title('Histogram for Total_Protiens Variable')
plt.show()


# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Total_Protiens', data = data,color='maroon')
plt.title('Boxplot for Total_Protiens Variable')
plt.show()

# %% [markdown]
# # Albumin

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Albumin'], kde = True,color='maroon')
plt.title('Histogram for Albumin Variable')
plt.show()


# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Albumin', data = data,color='maroon')
plt.title('Boxplot for Albumin Variable')
plt.show()

# %% [markdown]
# # Albumin_and_Globulin_Ratio

# %%
plt.figure(figsize=[12,4])
sns.histplot(data = data['Albumin_and_Globulin_Ratio'], kde = True,color='maroon')
plt.title('Histogram for Albumin_and_Globulin_Ratio Variable')
plt.show()


# %%
plt.figure(figsize=[12,3])
sns.boxplot(x = 'Albumin_and_Globulin_Ratio', data = data,color='maroon')
plt.title('Boxplot for Albumin_and_Globulin_Ratio Variable')
plt.show()

# %% [markdown]
# # Dataset

# %%
plt.figure(figsize=[12,4])
sns.countplot(y = data['Dataset'],palette='Greens')
plt.title('countplot for Dataset Variable')
plt.show()


# %% [markdown]
# # Bivariate Analysis

# %%
data.corr()

# %%
plt.figure(figsize = [20,8])
sns.heatmap(data.corr(),annot=True,cmap='magma', vmin=-1, vmax=1)

# %%
plt.figure(figsize=[12,4])
sns.countplot(x = data['Dataset'],hue=data['Gender'],palette='magma')
plt.title('countplot for Dataset Variable')
plt.show()


# %%
data.info()

# %%
var_list=['Total_Bilirubin', 'Direct_Bilirubin',
       'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
       'Albumin_and_Globulin_Ratio']
def draw_scattterplots(df, variables, n_rows, n_cols):
    fig=plt.figure(figsize = [20,20])
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        sns.scatterplot(x=df[var_name],y=df[var_name],ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

#test = pd.DataFrame(np.random.randn(30, 9), columns=map(str, range(9)))
draw_scattterplots(data, data[var_list],8,4)

# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Direct_Bilirubin'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin (Direct Vs Total)')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Alkaline_Phosphotase'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Alkaline_Phosphotase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Alamine_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Alamine_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Aspartate_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Aspartate_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Total_Protiens'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Total_Protiens')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Billirunin  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Direct_Bilirubin'],y=data['Alamine_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Direct_Bilirubin  Vs Alkaline_Phosphotase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Alamine_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Total_Bilirubin  Vs Alamine_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Aspartate_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Total_Bilirubin  Vs Aspartate_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Total_Bilirubin  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Alkaline_Phosphotase'],hue=data['Dataset'],palette='deep')
plt.title('Total_Bilirubin  Vs Alkaline_Phosphotase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Bilirubin'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Total_Bilirubin  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alkaline_Phosphotase'],y=data['Alamine_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Alkaline_Phosphotase  Vs Alamine_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alkaline_Phosphotase'],y=data['Aspartate_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Alkaline_Phosphotase  Vs Aspartate_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alkaline_Phosphotase'],y=data['Total_Protiens'],hue=data['Dataset'],palette='deep')
plt.title('Alkaline_Phosphotase  Vs Total_Protiens')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alkaline_Phosphotase'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Alkaline_Phosphotase  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alkaline_Phosphotase'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Alkaline_Phosphotase  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alamine_Aminotransferase'],y=data['Aspartate_Aminotransferase'],hue=data['Dataset'],palette='deep')
plt.title('Alamine_Aminotransferase  Vs Aspartate_Aminotransferase')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alamine_Aminotransferase'],y=data['Total_Protiens'],hue=data['Dataset'],palette='deep')
plt.title('Alamine_Aminotransferase  Vs Total_Protiens')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alamine_Aminotransferase'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Alamine_Aminotransferase  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Alamine_Aminotransferase'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Alamine_Aminotransferase  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Aspartate_Aminotransferase'],y=data['Total_Protiens'],hue=data['Dataset'],palette='deep')
plt.title('Aspartate_Aminotransferase  Vs Total_Protiens')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Aspartate_Aminotransferase'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Aspartate_Aminotransferase  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Aspartate_Aminotransferase'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Aspartate_Aminotransferase  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Protiens'],y=data['Albumin'],hue=data['Dataset'],palette='deep')
plt.title('Total_Protiens  Vs Albumin')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Total_Protiens'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Total_Protiens  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
plt.figure(figsize=[12,4])
sns.scatterplot(x = data['Albumin'],y=data['Albumin_and_Globulin_Ratio'],hue=data['Dataset'],palette='deep')
plt.title('Albumin  Vs Albumin_and_Globulin_Ratio')
plt.show()


# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Total_Bilirubin', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Total_Bilirubin', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Total_billirubin")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Direct_Bilirubin', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Direct_Bilirubin', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Direct_Bilirubin")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Alkaline_Phosphotase', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Alkaline_Phosphotase', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Alkaline_Phosphotase")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Alamine_Aminotransferase', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Alamine_Aminotransferase', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Alamine_Aminotransferase")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Aspartate_Aminotransferase', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Aspartate_Aminotransferase', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Aspartate_Aminotransferase")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Total_Protiens', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Total_Protiens', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Total_Protiens")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Albumin', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Albumin', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Albumin")
plt.show()

# %%
#plot 1:
plt.subplot(1, 2, 1)
sns.set_style("whitegrid")
sns.boxplot(x = 'Dataset', y = 'Albumin_and_Globulin_Ratio', data = data)



#plot 2:
plt.subplot(1, 2, 2)
sns.set_style("whitegrid")
sns.boxplot(x = 'Gender', y = 'Albumin_and_Globulin_Ratio', data = data)

plt.tight_layout()
plt.suptitle("Boxplot for Albumin_and_Globulin_Ratio")
plt.show()

# %% [markdown]
# # Machine Learning module

# %%
# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# %%
data["Albumin_and_Globulin_Ratio"] = data.Albumin_and_Globulin_Ratio.fillna(data['Albumin_and_Globulin_Ratio'].mean())
X = data.drop(['Dataset'], axis=1)

# %%
y = data['Dataset']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

# %%
#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))
print('Classification Report: \n', classification_report(y_test,log_predicted))

sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")

# %%

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
#Predict Output
gauss_predicted = gaussian.predict(X_test)

gauss_score = round(gaussian.score(X_train, y_train) * 100, 2)
gauss_score_test = round(gaussian.score(X_test, y_test) * 100, 2)
print('Gaussian Score: \n', gauss_score)
print('Gaussian Test Score: \n', gauss_score_test)
print('Accuracy: \n', accuracy_score(y_test, gauss_predicted))
print(confusion_matrix(y_test,gauss_predicted))
print(classification_report(y_test,gauss_predicted))

sns.heatmap(confusion_matrix(y_test,gauss_predicted),annot=True,fmt="d")

# %%
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
print('Random Forest Score: \n', random_forest_score)
print('Random Forest Test Score: \n', random_forest_score_test)
print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
print(confusion_matrix(y_test,rf_predicted))
print(classification_report(y_test,rf_predicted))

# %%



