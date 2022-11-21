# -*- coding:utf-8 -*-
import streamlit as st 
import streamlit.components.v1 as stc 

# import mini apps
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Application </h1>
		    <h4 style="color:white;text-align:center;">Diabetes </h4>
		</div>
		"""

dec_temp ="""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App"""

def main():
    stc.html(html_temp)
    st.title("Main App")

    menu = ["HOME", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "HOME":
        st.subheader("Home")
        st.markdown(dec_temp, unsafe_allow_html=True)
    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else: 
        st.subheader("About")


if __name__ == "__main__":
    main()