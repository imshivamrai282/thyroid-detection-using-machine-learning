
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
#import os

st.header("Thyroid Prediction App")
data = pd.read_csv('processed_data.csv')



# load model
best_xgboost_model = xgb.XGBClassifier()
best_xgboost_model.load_model('best_model.json')

if st.checkbox('Show Training Dataframe'):
    data

st.write("#")

st.subheader("Please select relevant features for your case")

age=st.number_input("Enter your Age: ", min_value=0,max_value=90, key="age",step=1)

left_column, right_column = st.columns(2)
sex=["Male","Female"]
tf=["Yes","No"]


gender = st.radio('Gender:',sex)
if(gender=="Male"):
    gender=1
else:
    gender=0


on_thyroxine=st.radio('Are you on thyroxine?',tf)
if(tf=="yes"):
    on_thyroxine=1
else:
    on_thyroxine=0

on_antithyroid = st.radio('Are you on antithyroid medication?', tf)
if(tf=="yes"):
    on_antithyroid=1
else:
    on_antithyroid=0

sick = st.radio('Are you sick?', tf)

if(tf=="yes"):
    sick=1
else:
    sick=0

preg=0
if (gender=="Female"):
    preg = st.radio('Are you pregnant?', tf)
    if (tf == "yes"):
        preg=1

surgery = st.radio('Have you undergone any thyroid related surgery?', tf)
if(tf=="yes"):
    surgery=1
else:
    surgery=0


i131_treat= st.radio('Are you undergoing I131 treatment?', tf)
if(tf=="yes"):
    i131_treat=1
else:
    i131_treat=0


hypothyroid = st.radio('Do you think you have hypothyroid?', tf)
if(tf=="yes"):
    hypothyroid=1
else:
    hypothyroid=0

hyperthyroid = st.radio('Do you think you have hyperthyroid?', tf)
if(tf=="yes"):
    hyperthyroid=1
else:
    hyperthyroid=0

goitre = st.radio('Do you have goitre?', tf)
if(tf=="yes"):
    goitre=1
else:
    goitre=0

tumor = st.radio('Do you have any kind of tumor(s)?', tf)
if(tf=="yes"):
    tumor=1
else:
    tumor=0

hypopituitary = st.radio('Do you have hyper pituitary gland?', tf)
if(tf=="yes"):
    hypopituitary=1
else:
    hypopituitary=0

psych = st.radio('Do you have any psych conditions/diseases(eg. Anxiety Disorders, Depression, PTSD, Eating Disorders, etc.) ?', tf)
if(tf=="yes"):
    psych=1
else:
    psych=0

st.write("#")

st.subheader("The following features must be selected after getting your blood test done")

tsh_measured=st.radio('Whether TSH was measured in the blood ?', tf)
if (tsh_measured=="Yes"):
    tsh=st.number_input("Enter TSH level in blood from lab work ", min_value=0.005,max_value=535.000, key="TSH",step=0.001)

st.write("#")

t3_measured=st.radio('Whether T3 was measured in the blood ?', tf)
if (t3_measured=="Yes"):
    t3=st.number_input("Enter T3 level in blood from lab work ", min_value=0.005,max_value=20.000, key="T3",step=0.001)

st.write("#")

t4_measured=st.radio('Whether T4 was measured in the blood ?', tf)
if (t4_measured=="Yes"):
    t4=st.number_input("Enter T4 level in blood from lab work ", min_value=0.000,max_value=700.000, key="T4",step=0.001)

st.write("#")

t4u_measured=st.radio('Whether T4U was measured in the blood ?', tf)
if (t4u_measured=="Yes"):
    t4u=st.number_input("Enter T4U level in blood from lab work ", min_value=0.100,max_value=3.000, key="T4U",step=0.001)

st.write("#")

fti_measured=st.radio('Whether FTI was measured in the blood ?', tf)
if (fti_measured=="Yes"):
    fti=st.number_input("Enter FTI level in blood from lab work ", min_value=0.000,max_value=900.000, key="FTI",step=0.001)

st.write("#")

global target
target=['A', 'AK', 'B', 'C', 'D', 'E', 'F', 'FK', 'G', 'GI', 'GK', 'GKJ', 'H', 'I', 'J', 'K', 'KJ', 'L', 'LJ', 'M', 'MI', 'MK', 'N', 'O', 'OI', 'P', 'Q', 'R', 'S', 'Z']
global target_value
target_value = {'A': 'Hyperthyroid conditions','B': 'T3 toxic levels','C': 'Toxic goitre conditions','D': 'Secondary toxic Hyperthyroid conditions','E': 'Hypothyroid conditions','F': 'Primary Hypothyroid','G': 'Compensated Hypothyroid','H': 'Secondary Hypothyroid','I': 'Increased in binding protein','J': 'Decreased in binding protein','K': 'Concurrent non-thyroidal illness','L': 'Consistent with replacement therapy','M': 'Replacement Therapy: Underreplaced','N': 'Replacement Therapy: Overreplaced','O': 'Antithyroid treatment suggested: Antithyroid Drugs','P': 'Antithyroid treatment: I131 treatment','Q': 'Antithyroid treatment: surgery','R': 'Discordant assay results','S': 'Elevated TBG','T': 'Elevated Thyroid Hormones','Z':'None'}


def res(pred):
    tar = target[pred[0]]
    length = len(tar)
    result=""
    if (length>1):
        #i = 2
        for i in range(length):
            value = target_value.get(tar[i])
            result+=value
            if (i!=length-1):
                result+=" with "
    else:
        result=target_value.get(tar)
    return result

if st.button('Make Prediction'):
    inputs =(age,gender,on_thyroxine,on_antithyroid,sick,preg,surgery,i131_treat, hypothyroid, hyperthyroid,goitre,tumor,hypopituitary,psych,tsh,t3,t4u,fti)
    #inputs= (27.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90.0, 0.4, 0.94, 7.5)
    n = np.asarray(inputs)
    r = n.reshape(1, -1)


    prediction = best_xgboost_model.predict(r)
    st.write("#")
    result=res(prediction)
    if result==None:
        st.subheader("No Hypothroid Conditions")
    else:
        st.subheader(result)










