from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Path setup
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "student_data.csv")

df = pd.read_csv("student_data.csv")
X, Y = df[['hours']], df['pass']

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=42,test_size=0.2)

#train model
model = LogisticRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(x_test,y_pred)

model_path = os.path.join(BASE_DIR,"model.pkl")
with open(model_path,"wb") as f:
    pickle.dump(model,f)
    
print("Model trained and saved!")