import sys, os, time, csv

k = int(sys.argv[1])

import pandas as pd
df1 = pd.read_csv('train_data.csv')
df2 = pd.read_csv('train_labels.csv')
print(df2.columns)

import pandas as pd
df1 = pd.read_csv('train_data.csv')
df2 = pd.read_csv('train_labels.csv')
df = pd.merge(df1, df2, on='Unnamed: 0')
df = df.drop(columns=['Unnamed: 0'])
df['Activity'].groupby(df['Activity']).count()

f = 'features.csv'
outfile = 'out.csv'

with open(f) as rd:
    if k == 1:
        ft = []
    else:
        ft = [x.strip() for x in rd.readlines()]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

X = df.drop(['Activity'],axis=1)
y = df['Activity']

def classify(clfname, clf):
    start_time = time.time()
    
    print(clfname, end='\t')
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    end_time = time.time()
    time_diff = end_time - start_time
    
    performance = {}
    
    performance['clfname'] = clfname
    performance['mrmr'] = k
    performance['shape'] = X.shape[1]
    performance['time'] = time_diff
    performance['train_accuracy'] = accuracy_score(y_train, y_train_pred) # Calculate Accuracy
    performance['train_precision'] = precision_score(y_train,y_train_pred,average = 'macro')
    performance['train_recall'] = recall_score(y_train,y_train_pred,average = 'macro')
    performance['train_mcc'] = matthews_corrcoef(y_train, y_train_pred) # Calculate MCC
    performance['train_f1'] = f1_score(y_train, y_train_pred, average='macro') # Calculate F1-score
    performance['test_accuracy'] = accuracy_score(y_test, y_test_pred) # Calculate Accuracy
    performance['test_precision'] = precision_score(y_test,y_test_pred,average = 'macro')
    performance['test_recall'] = recall_score(y_test,y_test_pred,average = 'macro')
    performance['test_mcc'] = matthews_corrcoef(y_test, y_test_pred) # Calculate MCC
    performance['test_f1'] = f1_score(y_test, y_test_pred, average='macro') # Calculate F1-score
    
    clf_out_data.append(performance)
    
    print(f'✓ ({time_diff} s)')

from mrmr import mrmr_classif

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

knn = KNeighborsClassifier(n_jobs=-1)
svm_rbf = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_jobs=-1)
mlp = MLPClassifier()

estimator_list = [('mlp', mlp), ('svm_rbf', svm_rbf), ('rf', rf), ('knn', knn)]

stack_model = StackingClassifier(estimators=estimator_list, final_estimator=LogisticRegression(solver='liblinear'))

classifiers = [knn, dt, rf, mlp, stack_model]
clf_names = ['knn', 'dt', 'rf', 'mlp', 'stack']

print(f'\nk = {k}')
selected_feature = mrmr_classif(X=X.drop(columns=ft), y=y, K=1)[0]
ft.append(selected_feature)
print('appended : ', selected_feature)

with open('features.csv', 'a') as sf:
    sf.write(selected_feature)
    sf.write('\n')

X1 = X[X.columns.intersection(ft)]

X_train, X_test, y_train, y_test = train_test_split(X1, y, stratify=y, test_size=0.2, random_state=42)

clf_out_data = []

perf_cols = ['clfname', 'mrmr', 'shape', 'time', 'train_accuracy', 'train_precision', 'train_recall', 'train_mcc', 'train_f1', 'test_accuracy', 'test_precision', 'test_recall', 'test_mcc', 'test_f1']

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

for clf in range(len(classifiers)):
    classify(clf_names[clf], classifiers[clf])

with open(outfile, mode='a', newline='') as file:
    writer = csv.writer(file)
    if os.stat(outfile).st_size == 0:
        writer.writerow(perf_cols)
    writer.writerows([entry.values() for entry in clf_out_data])
