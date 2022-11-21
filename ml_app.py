import streamlit as st 

import joblib
import os 

# eda 
import numpy as np

# URL : https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.

attrib_info = """
### Attribute Information
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle stiness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.
"""

label_dict = {"예": 1, "아니오" : 0}
gender_map = {"남성" : 1, "여성" : 0}
target_label_map = {"양성" : 1, "음성" : 0}

def get_fvalue(val):
    feature_dict = {"예": 1, "아니오": 0}
    for key, value in feature_dict.items():
        if val == key:
            return value 
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# ML 모델 불러오기
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def run_ml_app():
    st.subheader("ML")
    # st.write("It is working")
    # st.success("This is good.")

    with st.expander("Attribute Info"):
        st.markdown(attrib_info)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("나이(Age)", 10, 100)
        gender = st.radio("성별(Gender)", ["남성", "여성"])
        polyuria = st.radio("다뇨증(Polyuria)", ["예", "아니오"])
        polydipsia = st.radio("다음증(Polydipsia)", ["예", "아니오"])
        sudden_weight_loss = st.selectbox("급작스러운 체중 감소 (Sudden Weight Loss)", ["예", "아니오"])
        weakness = st.radio("쇠약(Weakness)", ["예", "아니오"])
        polyphagia = st.radio("다식증(Polyphagia)", ["예", "아니오"])
        genital_thrush = st.selectbox("구내염(Genital thrush)", ["예", "아니오"])

    with col2: 
        visual_blurring = st.selectbox("시야흐림(visual blurring)", ["예", "아니오"])
        itching = st.radio("가려움(Itching)", ["예", "아니오"])
        irritability = st.radio("과민성(Irritability)", ["예", "아니오"])
        delayed_healing = st.radio("지연치유(Delayed Healing)", ["예", "아니오"])
        partial_paresis = st.selectbox("부분 마비(Partial Paresis)", ["예", "아니오"])
        muscle_stiffness = st.radio("근육 뻣뻣함(Muscle Stiffness)", ["예", "아니오"])
        alopecia = st.radio("탈모(Alopecia)", ["예", "아니오"])
        obesity = st.select_slider("비만증(Obesity)", ["예", "아니오"])

    with st.expander("선택하신 내용"):
        result = {
            "age" : age,
            "gender" : gender,
            "polyuria" : polyuria,
            "polydipsia" : polydipsia, 
            "sudden_weight_loss" : sudden_weight_loss,
            "weakness" : weakness,
            "polyphagia" : polyphagia,
            "genital_thrush" : genital_thrush,
            "visual_blurring" : visual_blurring, 
            "itching" : itching, 
            "irritability" : irritability, 
            "delayed_healing" : delayed_healing, 
            "partial_paresis" : partial_paresis, 
            "muscle_stiffness" : muscle_stiffness, 
            "alopecia" : alopecia, 
            "obesity" : obesity
        }

        st.markdown("#### 입력결과 변환 전")
        st.write(result)

        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            elif i in ["남성", "여성"]:
                res = get_value(i, gender_map)
                encoded_result.append(res)
            else:
                encoded_result.append(get_fvalue(i))
        
        st.markdown("#### 입력결과 변환 후")
        st.write(encoded_result)

    with st.expander("예측 결과"):
        single_sample = np.array(encoded_result).reshape(1, -1)
        st.write(single_sample)

        model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")
        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)
        
        st.write(prediction)
        st.write(pred_prob)

        if prediction == 1:
            st.warning("양성입니다.")
            pred_proba_scores = {"양성일 확률": pred_prob[0][1]*100, "음성일 확률": pred_prob[0][0]*100}
            st.write(pred_proba_scores)
        else:
            st.success("음성입니다.")
            pred_proba_scores = {"양성일 확률": pred_prob[0][1]*100, "음성일 확률": pred_prob[0][0]*100}
            st.write(pred_proba_scores)





    
