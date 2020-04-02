#Importing the Libraries
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC

#Importing the dataset
dataset=pd.read_csv("student-mat.csv",sep=";")
original_data=dataset

#Spliting Datas into Training and Testing Sets
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=17)


#Confusion Matrix
def confuse(y_true, y_pred):
    cm = confusion_matrix(y_true,y_pred)
    print("\nConfusion Matrix: \n", cm)
    accuracy=accuracy_score(y_true,y_pred)
    print("This is the acurracy of model for y_true and y_predict  ::",accuracy)
    false_pass_rate(cm)
    false_fail_rate(cm)

#False Pass Rate 
def false_pass_rate(confusion_matrix):
    fp = confusion_matrix[0][1]#predict wrong pass
    tf = confusion_matrix[0][0]#predict true pass
    rate = float(fp) / (fp + tf)
    print("False Pass Rate: ", rate)

#False Fail Rate
def false_fail_rate(confusion_matrix):
    ff = confusion_matrix[1][0]#predict wrong fail
    tp = confusion_matrix[1][1]#predict true fail
    rate = float(ff) / (ff + tp)
    print("False Fail Rate: ", rate)

    #return rate

#Train Model and Print Score
def train_and_score(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    classifier = Pipeline([
        ('reduce_dim', SelectKBest(chi2, k=2)),
        ('train', LinearSVC(C=100))
    ])

    scores = cross_val_score(classifier, X_train, y_train, cv=5, n_jobs=2)
    #print("this are scores::",scores)
    print("Mean Model Accuracy:", np.array(scores).mean())

    classifier.fit(X_train, y_train)

    confuse(y_test, classifier.predict(X_test))
    #print("this is the result:",classifier.predict(X_test))






#Main Program 
def main():
    print("\nStudent Performance Prediction")

    # For each feature, encode to its  categorical values
    labelencoder = LabelEncoder()
    for column in dataset[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        dataset[column] = labelencoder.fit_transform(dataset[column].values)

    # Encode G1, G2, G3 as pass or fail binary values(pass=1)
    for i, row in dataset.iterrows():
        if row["G1"] >= 10:
            dataset.loc[dataset.G1][i] = 1
        else:
            dataset.loc[dataset.G1][i] = 0

        if row["G2"] >= 10:
            dataset["G2"][i] = 1
        else:
            dataset["G2"][i] = 0

        if row["G3"] >= 10:
            dataset["G3"][i] = 1
        else:
            dataset["G3"][i] = 0

    # Target values are G3
    y = dataset.pop("G3")
  
    # Feature set is remaining features
    X = dataset

    print("\n\nModel Accuracy Knowing G1 & G2 Scores")
    print("=====================================")
    train_and_score(X, y)

    # Removing  grade report 2
    X.drop(["G2"], axis = 1, inplace=True)
    print("\n\nModel Accuracy Knowing Only G1 Score")
    print("=====================================")
    train_and_score(X, y)

    # Removing grade report 1
    X.drop(["G1"], axis=1, inplace=True)
    print("\n\nModel Accuracy Without Knowing Scores")
    print("=====================================")
    train_and_score(X, y)



main()

    