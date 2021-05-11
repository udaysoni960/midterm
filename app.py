import streamlit as st
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("mid_term_1.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Dataset.csv')

# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, :-1].values
# Extracting dependent variable:
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.
imputer = imputer.fit(X[:, :])
#Replacing missing data with the calculated mean value
X[:, :]= imputer.transform(X[:,:])

# Splitting the Dataset into the Training set and Test set

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



def predict_note_authentication(Gender,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age):

  if(Gender=='Male'):
      ge=1
  else:
      ge=0


  output= model.predict(sc.transform([[ge,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age]]))
  print("Person will ",output)
  if output==[0]:
    prediction="Person is Diagnosed"


  if output==[1]:
    prediction="Person is not Diagnosed"


  print(prediction)
  return prediction
def main():

    html_temp = """
   <div class="" style="background-color:gray;" >
   <div class="clearfix">
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">MID TERM - I</p></center>
   <center><p style="font-size:30px;color:white;margin-top:10px;">NAME: UDAY SONI</p></center>
   <center><p style="font-size:25px;color:white;margin-top:0px;">-- PIET18CS147 -- Sec: C -- Roll No 35 --</p></center>
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Person Diagnose Prediction")
    Gender = st.selectbox('Insert Gender', ('Male', 'Female'))
    Glucose = st.number_input('Insert Glucose level')
    bp = st.number_input('Insert BP level')
    SkinThickness = st.number_input('Insert SkinThickness')
    Insulin = st.number_input('Insert Insulin level')
    BMI = st.number_input('Enter BMI')
    PedigreeFunction = st.number_input('Enter PedigreeFxn')
    Age = st.number_input('Enter Age')

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,bp,SkinThickness,Insulin,BMI,PedigreeFunction,Age)
      st.success('Model has predicted that -> {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by UDAY SONI")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
