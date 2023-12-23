#importing lib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk 

#%matplotlib
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

from sklearn.metrics import accuracy_score, classification_report

# reading data table
print("reading data")
print("-------------")

# Pandas read_csv() function imports a CSV file to DataFrame format.
drf = pd.read_csv('Cancer_Data.csv')

# A DataFrame is a data structure that organizes data into a 2-dimensional table of rows and columns, much like a spreadsheet. DataFrames are one of the most common data structures used in modern data analytics because they are a flexible and intuitive way of storing and working with data.
data =pd.DataFrame(drf)

print("table header:")
print("-------------")


# Definition and Usage. The head() method returns a specified number of rows, string from the top. The head() method returns the first 5 rows if a number is not specified. ;] Note: The column names will also be returned, in addition to the specified rows.
print(data.head())

print("-------------")
print("Data shape:")
print("-------------")


# Usually, on a broader scale, the shape() method is used to fetch the dimensions of Pandas and NumPy type objects in python. Every value represented by the tuple corresponds to the actual dimension in terms of array or row/columns.
print(data.shape)

print("============")
print("Data information:")

print("-------------")


# The info() method prints information about the DataFrame. The information contains the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values). Note: the info() method actually prints the info.
print(data.info())

print("============")
print("Data discription:")
print("-------------")


# The describe() method returns description of the data in the DataFrame. If the DataFrame contains numerical data, the description contains these information for each column: count - The number of not-empty values. mean - The average (mean) value. std - The standard deviation.
print(data.describe())

print("============")
print("diagnosis value counts:")
print("-------------")

# The function returns the count of all unique values in the given index in descending order, without any null values. The function returns the count of all unique values in the given index in descending order without any null values.
print(data.diagnosis.value_counts())

print("============")

# figure 1
print("Figure 1:")
print("Drowing value counts:")
print("-------------")
data.diagnosis.value_counts().plot(kind = "bar", color = ["salmon", "lightblue"])


# plt. show() starts an event loop, looks for all currently active figure objects, and opens one or more interactive windows that display your figure or figures.
plt.show()

print("*****************")

print("Sum of null data:")
print("-------------")

# The function dataframe. isnull(). sum(). sum() returns the number of missing values in the dataset.
print(data.isnull().sum())

print("look like the perfect dataset!!! No null values")
print("*****************")
# figure 2
print("Figure 2:")

# The purpose of using plt. figure() is to create a figure object. The whole figure is regarded as the figure object. It is necessary to explicitly use plt.
plt.figure(figsize = (10 , 8))


# Visualizing Data in Python Using plt.scatter() – Real Python
# A scatter plot is a visual representation of how two variables relate to each other. You can use scatter plots to explore the relationship between two variables, for example by looking for any correlation between them.
plt.scatter(data.radius_mean[data.diagnosis == 1], data.texture_mean[data.diagnosis == 1], color = "salmon")
plt.scatter(data.radius_mean[data.diagnosis == 0], data.texture_mean[data.diagnosis == 0], color = "lightblue")
plt.title("Cancer diagnosis")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend(["M=1", "B=0"])
plt.show()
print(" - - - - - - - - - -")
#Figure 3
print("Figure 3:")

plt.figure(figsize = (15 , 10))

# The Seaborn. heatmap() method is used to plot rectangular data in the form of a color-coded matrix. A heat map is one of the data visualization tools that shows the magnitude of a certain phenomenon in the form of colors.
sns.heatmap(data.corr(method='pearson'),annot=True)

plt.title("Figar 3:")
plt.show()
print("- =- =- =- =- =- =")

#Figure 4
print("Figure 4:")

# The drop() method removes the specified row or column. By specifying the column axis ( axis='columns' ), the drop() method removes the specified column.
x= data.drop('diagnosis',axis=1)
y = data['diagnosis']

# The corr() method finds the correlation of each column in a DataFrame.
x.corrwith(y).plot(kind = 'bar', grid = True,figsize=(12,8), title= "Figure 4: Correlation with diagnosis")

plt.show()
print("*=-=-=-=-=-=-=*")
#scaling and PCA
print("scaling and PCA :")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# StandardScaler removes the mean and scales each feature/variable to unit variance. This operation is performed feature-wise in an independent way. StandardScaler can be influenced by outliers (if they exist in the dataset) since it involves the estimation of the empirical mean and standard deviation of each feature.
scaler = StandardScaler()

# The fit_transform() method is used to fit the data into a model and transform it into a form that is more suitable for the model in a single step. This saves us the time and effort of calling both fit() and transform() separately.
X_standard = scaler.fit_transform(x)
print (X_standard)
print("+++++++++++++++")

#Reducing dimensionality
print("Reducing dimensionality :")
pca_model= PCA(n_components = 5)
pca_data_standard = pca_model.fit_transform(X_standard)
print(pca_data_standard)
print("$$$$$$$$$$$")

print("DATA Frame :")
dataset=pd.DataFrame(data = pca_data_standard, columns =['pc1', 'pc2', 'pc3', 'pc4' , 'pc5'])
dataset['diagnosis'] = y
print("Printing new DataSet header:")
print("----------")
print(dataset.head())
print("==========")
print(dataset.tail())
print("----------")

#Figure 5
print("Figure 5:")
x1= dataset.drop('diagnosis',axis=1)
y1 = dataset['diagnosis']
x1.corrwith(y1).plot(kind = 'bar', grid = True,figsize=(12,8), title= "Figure 5: Correlation with diagnosis")

plt.show()
print("*=-=-=-=-=-=-=*")

#Applying machine learning algorithms
print("Applying machine learning algorithms :")
from sklearn.model_selection  import  train_test_split
x= dataset.drop('diagnosis' , axis = 1)
y= dataset.diagnosis
print(y.shape)


# The train_test_split function is a powerful tool in Scikit-learn's arsenal, primarily used to divide datasets into training and testing subsets. This function is part of the sklearn. model_selection module, which contains utilities for splitting data.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state=42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print("=-=-=-=-=-=-=")

print("Decision Tree Classifier :")
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeClassifier is a class capable of performing multi-class classification on a dataset. In case that there are multiple classes with the same and highest probability, the classifier will predict the class with the lowest index amongst those classes.
tree_clf= DecisionTreeClassifier(random_state = 42)

# X_train and y_train sets are used for training and fitting the model. The X_test and y_test sets are used for testing the model if it's predicting the right outputs/labels. we can explicitly test the size of the train and test sets. It is suggested to keep our train sets larger than the test sets.
tree_clf.fit(x_train, y_train)

# predict() : given a trained model, predict the label of a new set of data. This method accepts one argument, the new data X_new (e.g. model. predict(X_new) ), and returns the learned label for each object in the array.
pred = tree_clf.predict(x_test)


# The accuracy_score function computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions. In multilabel classification, the function returns the subset accuracy.
print("Accuracy Score: ", accuracy_score(y_test, pred) * 100)

# The classification report shows a representation of the main classification metrics on a per-class basis. This gives a deeper intuition of the classifier behavior over global accuracy which can mask functional weaknesses in one class of a multiclass problem.
print("Classification Report: \n", classification_report (y_test, pred))

print("=-=-=-=-=-=-=")
#K-nearest neighbor
print("K-nearest neighbor :")

# KNeighborsClassifier is a supervised learning algorithm that makes classifications based on data neighbors.
# Like? Let's take one more example: Suppose we have a sample X (in this case, it's the green dot).
from sklearn.neighbors import KNeighborsClassifier
knn_clf= KNeighborsClassifier()
knn_clf.fit(x_train, y_train)
pred = knn_clf.predict(x_test)

# The accuracy_score function of the sklearn.
# metrics package calculates the accuracy score for a set of predicted labels against the true labels.
print("Accuracy Score: ", accuracy_score(y_test, pred) * 100)
print("Classification Report: \n", classification_report (y_test, pred))
# End