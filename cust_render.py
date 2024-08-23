import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as pt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder



st.title("Customer conversion prediction - Insurance")
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

with st.sidebar:
   option=st.selectbox("SELECT ONE:",("ANALYSIS","PREDICTION"),index=None,placeholder=" ")

if option=="ANALYSIS":
    st.write("AGE ANALYSIS")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='age', data=df_new1,palette='viridis')
    plt.title('AGE VS COUNT')
    plt.xlabel('age')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('People aged between 30-60 are targetted more. ')

    st.write("AGE VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['age'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='age',y='y1',data=ss1,palette='viridis')
    plt.title('AGE VS OUTCOME')
    plt.xlabel('age')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,older people(80-95) took insurance more compared to younger aged people')

    # job
    st.write("JOB ANALYSIS")
    f1=plt.figure(figsize=(15,8))
    sns.countplot(x='job',data=df_new1,palette='pastel')
    plt.title('JOB vs COUNT')
    plt.xticks(rotation=90)
    st.pyplot(f1)
    st.write("MANAGEMENT AND BLU COLLAR PPL were targetted more")

    st.write("JOB VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['job'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='job',y='y1',data=ss1,palette='pastel')
    plt.title('JOB VS OUTCOME')
    plt.xlabel('job')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,students and retired ppl took insurance more compared to others')


    st.write("MARITAL ANALYSIS ")
    f=plt.figure(figsize=(15,8))
    palette=['blue', 'green', 'red', 'orange', 'purple']
    sns.countplot(x='marital', data=df_new1,palette=palette)
    plt.title('MARITAL VS COUNT')
    plt.xlabel('marital')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write("Married ppl were targetted more compared to single and divorced")

    st.write("MARITAL VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['marital'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    p=['blue', 'green', 'red', 'orange', 'purple']
    sns.barplot(x='marital',y='y1',data=ss1,palette=p)
    plt.title('MARITAL VS OUTCOME')
    plt.xlabel('MARITAL')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,single ppl took insurance more compared to others')


    st.write("CALL TYPE ANALYSIS ")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='call_type', data=df_new1,palette='dark')
    plt.title('CALL TYPE COUNT')
    plt.xlabel('call_type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(f)

    st.write("CALL TYPE VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['call_type'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='call_type',y='y1',data=ss1,palette='dark')
    plt.title('CALL TYPE VS OUTCOME')
    plt.xlabel('CALL TYPE')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,ppl used cellphones took insurance more compared to others')



    st.write("NUMBER OF CALLS ANALYSIS ")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='num_calls',data=df_new1,palette='muted')
    plt.title('NUMBER OF CALLS MADE')
    plt.xlabel('num_calls')
    plt.ylabel('Outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write("Mostly number of calls made once only. Average number of calls made 3.")

    st.write("NUMBER OF CALLS VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['num_calls'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='num_calls',y='y1',data=ss1,palette='muted')
    plt.title('NUM_CALLS VS OUTCOME')
    plt.xlabel('NUM_CALLS')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,Most ppl took insurance in one call only')


    st.write("MONTH ANALYSIS")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='mon',data=df_new1,palette='pastel')
    plt.title('MONTH VS CONVERSION COUNT')
    plt.xlabel('mon')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write("May month has been targetted more")

    st.write("MONTH VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['mon'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='mon',y='y1',data=ss1,palette='pastel')
    plt.title('MONTH VS OUTCOME')
    plt.xlabel('MONTH')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,ppl took more insurance on march,sept,october and december')

    st.write("PREVIOUS OUTCOME ANALYSIS ")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='prev_outcome',data=df_new1,palette='Set1')
    plt.title('PREVIOUS OUTCOME VS COVERSION COUNT')
    plt.xlabel('prev_outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write("Most of the previous outcome of the calls are unknown.")

    st.write("PREVIOUS OUTCOME VS OUTCOME ANALYSIS")
    f=plt.figure(figsize=(15,8))
    ss1=df_new1.groupby(['prev_outcome'])[['y1']].mean()
    ss1.reset_index(inplace=True)
    sns.barplot(x='prev_outcome',y='y1',data=ss1,palette='Set1')
    plt.title('PREV_OUTCOME VS OUTCOME')
    plt.xlabel('PREV_OUTCOME')
    plt.ylabel('outcome')
    plt.xticks(rotation=90)
    st.pyplot(f)
    st.write('Based on the plot,If the previous outcome of the call is success, then the customer is more likely to convert.')

    st.write("OUTCOME ANALYSIS ")
    f=plt.figure(figsize=(15,8))
    sns.countplot(x='Outcome',data=df_new1,palette='dark')
    plt.title('OUTCOME')
    plt.xlabel('OUTCOME')
    plt.ylabel('COUNT')
    plt.xticks(rotation=90)
    st.pyplot(f)

    st.write("AGE VS JOB VS MARITAL VS NUM_CALLS VS OUTCOME ANALYSIS")
    fig=sns.relplot(df_new1, x="job", y="age", hue="marital",col='Outcome',size='num_calls',col_wrap=1,sizes=(10,100))
    plt.xticks(rotation=90)
    st.pyplot(fig)

elif option=='PREDICTION':
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
            submit_bt = st.form_submit_button(label='Predict Customer Conversion',use_container_width=150)
            st.markdown('''
                ''', unsafe_allow_html=True)

            if submit_bt:
                with open(r'adaboost_pkl','rb') as f:
                    model=pickle.load(f)
                    #print(job,mar_en,edu_en,call_en,d_en,mon_en,ag_en,dur_en,num_en,pr_o_en)
                    data = np.array([j_dict[job], 
                                    mar_dict[mar_st],
                                    e_dict[edu],
                                    c1_dict[call],
                                    d_en,
                                    mon_dict[mon_en],
                                    np.log(float(ag_en)), 
                                    np.log(float(dur_en)),
                                    np.log(float(num_en)),
                                    prev_dict[pr_o]]).reshape(1,-1)
                    
                    print(data)
                    y_pred = model.predict(data)
                    print(y_pred[0])
                    #inverse transformation 
                    converse_bfr = np.exp(y_pred[0])
                    converse_aft = np.round(converse_bfr,2)
                    if converse_aft==1.0:
                        st.write(f"WHETHER THIS CUSTOMER WILL CONVERT: YES ")
                    else:
                        st.write(f"WHETHER THIS CUSTOMER WILL CONVERT: NO ")
                        

