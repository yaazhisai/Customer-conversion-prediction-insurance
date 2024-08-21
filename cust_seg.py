import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder



st.title("Customer Segmentation prediction - Insurance")
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color:black;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
	<style>
	.stSelectbox:first-of-type > div[data-baseweb="select"] > div {
	      background-color:grey;
    	      padding: 10px;
	}
	</style>
""", unsafe_allow_html=True)

df_new1=pd.read_csv('df.csv')

with st.form("my-form"):
        j=df_new1['job'].unique()
        j_dict={'management':4., 'technician':9., 'entrepreneur':2., 'blue-collar':1.,'unknown':11., 'retired':5., 'admin.':0., 'services':7., 'self-employed':6.,
       'unemployed':10., 'housemaid':3., 'student':8.}
        mar=df_new1['marital'].unique()
        mar_dict={'married':1, 'single':2, 'divorced':0}
        
        e=df_new1['education_qual'].unique()
        e_dict={'tertiary':2., 'secondary':1., 'unknown':3., 'primary':0.}
        c1=df_new1['call_type'].unique()
        c1_dict={'unknown':2, 'cellular':0, 'telephone':1}
        d=df_new1['day'].unique()
        mon=['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan','feb','mar', 'apr', 'sep']
        mon_dict={'jan':1,"feb":2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        age=df_new1['age'].unique()
        dur=df_new1['dur'].unique()
        num=df_new1['num_calls'].unique()
        prev=df_new1['prev_outcome'].unique()
        prev_dict={'unknown':3, 'failure':0, 'other':1, 'success':2}

        col1, col2, col3 = st.columns([5, 1,5])
        with col1:
            st.write(' ')
            job=st.selectbox('JOB',j,help='Job role of the customer.')
            mar_st=st.selectbox('MARITAL',mar,help='Marital status of the customer')
            edu=st.selectbox('EDUCATION',e,help='Education level of the customer')
            call=st.selectbox('CALL TYPE',c1,help='Type of call made')
            d1=st.selectbox('DAY',d,help='Day of the week the call made')
            d_en=df_new1[df_new1['day']==d1]['day'].iloc[0]
            
            
        with col3:
            st.write('  ')
            mon_en=st.selectbox('MONTH',mon,help='Month of the year.')
            ag=st.selectbox('AGE',age,help='Age of the customer')
            ag_en=df_new1[df_new1['age']==ag]['age'].iloc[0]
            dur=st.selectbox('CALL DURATION',dur,help='Duration of the call.')
            dur_en=df_new1[df_new1['dur']==dur]['dur'].iloc[0]
            num=st.selectbox('NO.OF CALLS',num,help=' Number of calls made to the customer.')
            num_en=df_new1[df_new1['num_calls']==num]['num_calls'].iloc[0]
            pr_o=st.selectbox('PREV OUTCOME',prev,help='Outcome of the previous call.')
            st.write(' ')
            st.write('  ')
            submit_bt = st.form_submit_button(label='SEGMENT CUSTOMER TO CLUSTERS',use_container_width=150)
            st.markdown('''
                ''', unsafe_allow_html=True)

            if submit_bt:
                with open(r'C:\Users\yaazhisai\Desktop\Final capstone\clustm_pkl','rb') as f:
                    model=pickle.load(f)
                    #print(job,mar_en,edu_en,call_en,d_en,mon_en,ag_en,dur_en,num_en,pr_o_en)
                    data = np.array([j_dict[job], 
                                    mar_dict[mar_st],
                                    e_dict[edu],
                                    c1_dict[call],
                                    d_en,
                                    mon_dict[mon_en],
                                    ag_en, 
                                    dur_en,
                                    num_en,
                                    prev_dict[pr_o]]).reshape(1,-1)
                    
                    print(data)
                    y_pred = model.predict(data)
                    st.write(f"THIS CUSTOMER IS  SEGMENTED INTO CLUSTER: ",y_pred)
                
                        



