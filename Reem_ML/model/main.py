import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_decision_tree_model(data , criterion):
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']

  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  # train the model
  model = DecisionTreeClassifier(criterion = criterion, max_depth=3, random_state=0)
  model.fit(X_train, y_train)

  # test model
  y_pred = model.predict(X_test)
  print(f'Accuracy of the Decision Tree model with {criterion} criterion:  ', accuracy_score(y_test, y_pred))
  print(f'Classification report of the Decision Tree model with {criterion} criterion: \n', classification_report(y_test, y_pred),"\n\n")

  # cm = confusion_matrix(y_test, y_pred)
  # print('Confusion matrix\n\n', cm)
  #
  # plt.figure(figsize=(14, 10))
  # tree.plot_tree(model)
  # plt.show()
  return model, scaler



def create_Random_Forest_model(data ,n_estimators ):
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']

  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  # train the model
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=0 )
  model.fit(X_train, y_train)

  # test model
  y_pred = model.predict(X_test)
  print(f'Accuracy of the Random Forest model with {n_estimators} estimators :', accuracy_score(y_test, y_pred))
  print(f'Classification report of the Random Forest model with {n_estimators} estimators : \n', classification_report(y_test, y_pred),"\n\n")

  return model, scaler


def create_logistic_regression_model(data):
  X = data.drop(['diagnosis'], axis=1)
  y = data['diagnosis']

  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  # train the model
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # test model
  y_pred = model.predict(X_test)
  print('Accuracy of the logistic regression model: ', accuracy_score(y_test, y_pred))
  print("Classification report of the logistic regression model: \n", classification_report(y_test, y_pred),"\n\n")

  return model, scaler



def get_clean_data():
  url = "https://raw.githubusercontent.com/mhammoud14/ML_Project/main/data.csv"
  data = pd.read_csv(url)

  print("The column Unnamed is a useless column that will not have an effect on the prediction , thus it should be dumped")
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def frequency_distribution(data):
  print("The frequency of the values in each column are :")
  for var in data.columns:
    print(data[var].value_counts())

def check_missing_values(data):
  print("\nThe missing values in each column are as follows :")
  print(data.isnull().sum())
  print("\nAs we can see , we have no missing values, thus we are ready to train the model\n\n")




def main():
  data = get_clean_data()
  # frequency_distribution(data)
  # check_missing_values(data)

  create_decision_tree_model(data , "gini")
  create_decision_tree_model(data , "entropy")
  create_Random_Forest_model(data, 10)
  create_Random_Forest_model(data, 100)
  model , scaler = create_logistic_regression_model(data)

  joblib.dump(model, 'model.sav')
  joblib.dump(scaler, 'scaler.sav')


  # with open('model/model.pkl', 'wb') as f:
  #   pickle.dump(model, f)
  #
  # with open('model/scaler.pkl', 'wb') as f:
  #   pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()