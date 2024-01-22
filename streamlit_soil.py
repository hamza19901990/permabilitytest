import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np
import pandas as pd
import csv
import streamlit as st
from PIL import Image

st.write("""
# Concrete Compressive Strength Prediction
This app predicts the **Unconfied Compressive Strength (UCS) of Geopolymer Stabilized Clayey Soil**!
""")
st.write('---')
image=Image.open(r'Unconfined-Compressive-Strength-Test-Apparatus.jpg')
st.image(image, use_column_width=True)

data = pd.read_csv(r"soilnew.csv")

req_col_names = ["d10 (mm)", "d50 (mm)", "d60 (mm)", "e","k (m/s)"]
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)
st.subheader('data information')
data.head()
data.isna().sum()
corr = data.corr()
st.dataframe(data)

X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]          # Target - Last Column
print(X)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
st.sidebar.header('Specify Input Parameters')
"d10 (mm)", "d50 (mm)", "d60 (mm)", "e"
def get_input_features():
    d10 (mm) = st.sidebar.slider('d10', 14.07,88.46,15.55)
    d50 (mm) = st.sidebar.slider('d50',0,50,25)
    d60 (mm) = st.sidebar.slider('d60', 0,20,15)
    e = st.sidebar.slider('air void', 4,15,6)


    data_user = {'d10': p_index,
            'd50': ggbfs_per,
            'd60': fly_perc,
            'M': M,
             'air void': e,

    }
    features = pd.DataFrame(data_user, index=[0])
    return features

df = get_input_features()
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')




# Reads in saved classification model
import pickle
load_clf = pickle.load(open('ada_boost_model.pkl', 'rb'))
st.header('Prediction of UCS (Mpa)')

# Apply model to make predictions
prediction = load_clf.predict(df)
st.write(prediction)
st.write('---')
