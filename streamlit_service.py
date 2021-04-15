"""
To run this app, in your terminal:
> streamlit run streamlit_service.py

Source: https://is.gd/SobJvL
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from itertools import chain
#preprocessing 

def convertAge(string_age):
    if string_age =="" :
        return "unknown_age"
    if 0 <= float(string_age)<=2:
        return "0-2 yr"
    if 3 <= float(string_age)<=12:
        return "3-12 yr"
    if 13 <= float(string_age)<=19:
        return "13-19 yr"
    if 20 <= float(string_age)<=39:
        return "20-39 yr"
    if 40 <= float(string_age)<=59:
        return "40-59 yr"
    return "over_60 yr"

# Loading model 
clf_mo = joblib.load('./model/kmeans_MODERNA_model.joblib')
clf_pf = joblib.load('./model/kmeans_PFIZER_model.joblib')
ag_input_tansformer_mo = joblib.load('./model/onehotendcoder_moderna.joblib')
hc_input_tansformer_mo = joblib.load('./model/tfidf_vectorizer_moderna.joblib')
ag_input_tansformer_pf = joblib.load('./model/onehotendcoder_pfizer.joblib')
hc_input_tansformer_pf = joblib.load('./model/tfidf_vectorizer_pfizer.joblib')

# Create title and sidebar
st.title("Vaccine-Symptoms Predictor")
st.sidebar.title("Please provide the information below.")


# Intializing parameter values
parameter_list=['Age','Gender','Health Condition']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]

# Display above values in the sidebar
#for parameter, parameter_df in zip(parameter_list, parameter_default_values):
age_input= st.sidebar.slider(label='Age', key='Age',value=int(47), min_value=0, max_value=120, step=1)
gender_input = st.sidebar.selectbox("Biological Gender", ['Female', 'Male', "Not Specified"])
hc_input= st.sidebar.text_input("Exisiting Health Condition", 'disease1, disease2, ...')


#process age and gender
sex = ""
if(gender_input == "Female"):
	sex = "F"
if(gender_input == "Male"):
	sex = "M"
if(gender_input == "Not specified"):
	sex = "U"



agecat = convertAge(age_input)
new_sample = pd.DataFrame({ 'sex':[sex], 'age_cat':[agecat] }, columns= ['age_cat','sex'] )


#Moderna
sample_dummi_MODERNA =ag_input_tansformer_mo.transform(new_sample[['sex','age_cat']])
sample_dummi_MODERNA =pd.DataFrame(sample_dummi_MODERNA.toarray() , columns=ag_input_tansformer_mo.get_feature_names())
#parameter_input_values.append(hc_input)

#process health condition string
sample_tfidf_MODERNA=hc_input_tansformer_mo.transform([hc_input])
sample_tfidf_MODERNA=pd.DataFrame(sample_tfidf_MODERNA.toarray(), columns=hc_input_tansformer_mo.get_feature_names())

#predict symptoms
sampleData_MODERNA=pd.concat([sample_dummi_MODERNA ,sample_tfidf_MODERNA], axis=1)



#Pfizer
sample_dummi_Pfizer =ag_input_tansformer_pf.transform(new_sample[['sex','age_cat']])
sample_dummi_Pfizer =pd.DataFrame(sample_dummi_Pfizer.toarray() , columns=ag_input_tansformer_pf.get_feature_names())
#parameter_input_values.append(hc_input)

#process health condition string
sample_tfidf_Pfizer=hc_input_tansformer_pf.transform([hc_input])
sample_tfidf_Pfizer=pd.DataFrame(sample_tfidf_Pfizer.toarray(), columns=hc_input_tansformer_pf.get_feature_names())

#predict symptoms
sampleData_Pfizer=pd.concat([sample_dummi_Pfizer ,sample_tfidf_Pfizer], axis=1)




#input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')



# Button that triggers the actual prediction
if st.button("Click Here to Predict"):
	prediction1 = clf_mo.predict(sampleData_MODERNA)
	prediction2 = clf_pf.predict(sampleData_Pfizer)



	df_Moderna = pd.read_pickle("./model/MODERNA_DF.pkl")
	df_Pfizer = pd.read_pickle("./model/PFIZER_DF.pkl")
	
	#moderna
	df1 = df_Moderna[df_Moderna['CLUSTER'] == 3]
	mo_sy = df1['SYMPTOMS'].head(10).tolist()
	
	mo_sy = list(chain.from_iterable(mo_sy))
	mo_res = []
	for i in mo_sy:
		if i not in mo_res:
			mo_res.append(i)
	
	#pfizer
	df1 = df_Pfizer[df_Pfizer['CLUSTER'] == 3]
	pf_sy = df1['SYMPTOMS'].head(10).tolist()
	
	pf_sy = list(chain.from_iterable(pf_sy))
	pf_res = []
	for i in pf_sy:
		if i not in pf_res:
			pf_res.append(i)
	# Display the corresponding image based on the prediction made by the model
	
	"Top 10 side-effects you may experience if you choose Moderna vaccine:"
	mo_res[:10]
	"Top 10 side-effects you may experience if you choose Pfizer vaccine:"
	pf_res[:10]
