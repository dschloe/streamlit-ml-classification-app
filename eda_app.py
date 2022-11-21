import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import plotly.express as px 

# 데이터 불러오기
@st.cache 
def load_data(data):
    df = pd.read_csv(data)
    return df 

def run_eda_app():
    st.subheader("EDA")
    # df = pd.read_csv("data/diabetes.csv")
    df = load_data("data/diabetes.csv")
    # st.dataframe(df)

    submenu = st.sidebar.selectbox("Submenu", ['기술통계량', '그래프'])
    if submenu == '기술통계량':
        st.dataframe(df)

        with st.expander("Data Types"):
            df2 = pd.DataFrame(df.dtypes).transpose()
            df2.index = ['구분']
            st.dataframe(df2)
        
        with st.expander("Descriptive Summary"):
            st.dataframe(pd.DataFrame(df.describe()).transpose())
        
        with st.expander("Class Distribution"):
            st.dataframe(df['class'].value_counts())

        with st.expander("Gender Distribution"):
            st.dataframe(df['Gender'].value_counts()) 
        
    elif submenu == "그래프":
        st.subheader('Plots')

        # Layouts
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.expander("Dist Plot"):

                # Seaborn
                fig, ax = plt.subplots()
                df2 = sns.load_dataset("titanic")
                sns.countplot(x=df2["class"])
                # sns.countplot(df['Gender'], ax=ax)
                st.pyplot(fig)

                gen_df = df['Gender'].value_counts()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type', 'Counts']
                st.dataframe(gen_df)

        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(gen_df)

            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())

        # Freq Dist
        with st.expander("Frequency Distribution"):
            freq_df = df['Age'].value_counts()
            p2 = px.bar(freq_df, x = freq_df.index, y = freq_df.values)
            st.plotly_chart(p2)
         

        # Outlier Detection
        with st.expander("Outlier Detection Plot"):
            fig, ax = plt.subplots()
            sns.boxplot(df['Age'], ax=ax)
            st.pyplot(fig)

            p3 = px.box(df, x='Age', color = 'Gender')
            st.plotly_chart(p3)

       
            


        

                