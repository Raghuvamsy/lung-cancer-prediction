# Lung Cancer Prediction Using 10 models

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ignoring warning
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'C:\GitHub\lung-cancer-prediction/survey lung cancer.csv')

# Data inspection
df.shape
df.duplicated().sum()
df = df.drop_duplicates()
df.isnull().sum()
df.info()
df.describe()

# Label Encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
df['SMOKING'] = le.fit_transform(df['SMOKING'])
df['YELLOW_FINGERS'] = le.fit_transform(df['YELLOW_FINGERS'])
df['ANXIETY'] = le.fit_transform(df['ANXIETY'])
df['PEER_PRESSURE'] = le.fit_transform(df['PEER_PRESSURE'])
df['CHRONIC DISEASE'] = le.fit_transform(df['CHRONIC DISEASE'])
df['FATIGUE '] = le.fit_transform(df['FATIGUE '])
df['ALLERGY '] = le.fit_transform(df['ALLERGY '])
df['WHEEZING'] = le.fit_transform(df['WHEEZING'])
df['ALCOHOL CONSUMING'] = le.fit_transform(df['ALCOHOL CONSUMING'])
df['COUGHING'] = le.fit_transform(df['COUGHING'])
df['SHORTNESS OF BREATH'] = le.fit_transform(df['SHORTNESS OF BREATH'])
df['SWALLOWING DIFFICULTY'] = le.fit_transform(df['SWALLOWING DIFFICULTY'])
df['CHEST PAIN'] = le.fit_transform(df['CHEST PAIN'])
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

df.info()

# Target Distribution
target_plot = sns.countplot(x='LUNG_CANCER', data=df)
plt.title('Target Distribution')
plt.show()
df['LUNG_CANCER'].value_counts()

# Data Visualization

def plot(col, df=df):
    return df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8,5))

plot('GENDER')
plot('AGE')
plot('SMOKING')
plot('YELLOW_FINGERS')
plot('ANXIETY')
plot('PEER_PRESSURE')
plot('CHRONIC DISEASE')
plot('FATIGUE ')
plot('ALLERGY ')
plot('WHEEZING')
plot('ALCOHOL CONSUMING')
plot('COUGHING')
plot('SHORTNESS OF BREATH')
plot('SWALLOWING DIFFICULTY')
plot('CHEST PAIN')

# Drop less relevant features
df_new = df.drop(columns=['GENDER', 'AGE', 'SMOKING', 'SHORTNESS OF BREATH'])

# Correlation
cn = df_new.corr()
cmap = sns.diverging_palette(260, -10, s=50, l=75, n=6, as_cmap=True)
plt.subplots(figsize=(18, 18))
sns.heatmap(cn, cmap=cmap, annot=True, square=True)
plt.show()
kot = cn[cn >= .40]
plt.figure(figsize=(12, 8))
sns.heatmap(kot, cmap="Blues")
plt.show()

# Feature Engineering
df_new['ANXYELFIN'] = df_new['ANXIETY'] * df_new['YELLOW_FINGERS']

# Splitting independent and dependent variables
X = df_new.drop('LUNG_CANCER', axis=1)
y = df_new['LUNG_CANCER']

# Target Distribution Imbalance Handling
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score, f1_score
lr_cr = classification_report(y_test, y_lr_pred)
print(lr_cr)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_dt_pred = dt_model.predict(X_test)
dt_cr = classification_report(y_test, y_dt_pred)
print(dt_cr)

# K Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)
knn_cr = classification_report(y_test, y_knn_pred)
print(knn_cr)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
y_gnb_pred = gnb_model.predict(X_test)
gnb_cr = classification_report(y_test, y_gnb_pred)
print(gnb_cr)

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)
y_mnb_pred = mnb_model.predict(X_test)
mnb_cr = classification_report(y_test, y_mnb_pred)
print(mnb_cr)

# Support Vector Classifier
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_svc_pred = svc_model.predict(X_test)
svc_cr = classification_report(y_test, y_svc_pred)
print(svc_cr)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)
rf_cr = classification_report(y_test, y_rf_pred)
print(rf_cr)

# XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_xgb_pred = xgb_model.predict(X_test)
xgb_cr = classification_report(y_test, y_xgb_pred)
print(xgb_cr)

# Multi-layer Perceptron classifier
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)
y_mlp_pred = mlp_model.predict(X_test)
mlp_cr = classification_report(y_test, y_mlp_pred)
print(mlp_cr)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_gb_pred = gb_model.predict(X_test)
gb_cr = classification_report(y_test, y_gb_pred)
print(gb_cr)

# K-Fold Cross Validation
from sklearn.model_selection import KFold, cross_val_score
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
lr_model_scores = cross_val_score(lr_model, X, y, cv=kf)
dt_model_scores = cross_val_score(dt_model, X, y, cv=kf)
knn_model_scores = cross_val_score(knn_model, X, y, cv=kf)
gnb_model_scores = cross_val_score(gnb_model, X, y, cv=kf)
mnb_model_scores = cross_val_score(mnb_model, X, y, cv=kf)
svc_model_scores = cross_val_score(svc_model, X, y, cv=kf)
rf_model_scores = cross_val_score(rf_model, X, y, cv=kf)
xgb_model_scores = cross_val_score(xgb_model, X, y, cv=kf)
mlp_model_scores = cross_val_score(mlp_model, X, y, cv=kf)
gb_model_scores = cross_val_score(gb_model, X, y, cv=kf)
print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Decision tree models' average accuracy:", np.mean(dt_model_scores))
print("KNN models' average accuracy:", np.mean(knn_model_scores))
print("Gaussian naive bayes models' average accuracy:", np.mean(gnb_model_scores))
print("Multinomial naive bayes models' average accuracy:", np.mean(mnb_model_scores))
print("Support Vector Classifier models' average accuracy:", np.mean(svc_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))
print("XGBoost models' average accuracy:", np.mean(xgb_model_scores))
print("Multi-layer perceptron models' average accuracy:", np.mean(mlp_model_scores))
print("Gradient boost models' average accuracy:", np.mean(gb_model_scores))

# Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=k)
lr_model_scores = cross_val_score(lr_model, X, y, cv=kf)
dt_model_scores = cross_val_score(dt_model, X, y, cv=kf)
knn_model_scores = cross_val_score(knn_model, X, y, cv=kf)
gnb_model_scores = cross_val_score(gnb_model, X, y, cv=kf)
mnb_model_scores = cross_val_score(mnb_model, X, y, cv=kf)
svc_model_scores = cross_val_score(svc_model, X, y, cv=kf)
rf_model_scores = cross_val_score(rf_model, X, y, cv=kf)
xgb_model_scores = cross_val_score(xgb_model, X, y, cv=kf)
mlp_model_scores = cross_val_score(mlp_model, X, y, cv=kf)
gb_model_scores = cross_val_score(gb_model, X, y, cv=kf)
print("Logistic regression models' average accuracy:", np.mean(lr_model_scores))
print("Decision tree models' average accuracy:", np.mean(dt_model_scores))
print("KNN models' average accuracy:", np.mean(knn_model_scores))
print("Gaussian naive bayes models' average accuracy:", np.mean(gnb_model_scores))
print("Multinomial naive bayes models' average accuracy:", np.mean(mnb_model_scores))
print("Support Vector Classifier models' average accuracy:", np.mean(svc_model_scores))
print("Random forest models' average accuracy:", np.mean(rf_model_scores))
print("XGBoost models' average accuracy:", np.mean(xgb_model_scores))
print("Multi-layer perceptron models' average accuracy:", np.mean(mlp_model_scores))
print("Gradient boost models' average accuracy:", np.mean(gb_model_scores))
