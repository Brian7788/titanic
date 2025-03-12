import streamlit as st
import joblib
import pandas as pd

# 加载模型
model = joblib.load("model.pkl")

st.title("Titanic Survival Prediction")  # 设置标题

# 创建输入框
pclass = st.number_input("Pclass (1 = First, 2 = Second, 3 = Third)", min_value=1, max_value=3, step=1)
sex = st.radio("Sex", options=["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("SibSp (Number of Siblings/Spouses)", min_value=0, max_value=10, step=1)
parch = st.number_input("Parch (Number of Parents/Children)", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, step=0.1)
embarked = st.radio("Embarked", options=["C", "Q", "S"])

# 把分类数据转换成数值
sex = 1 if sex == "Male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked = embarked_dict[embarked]

# 预测按钮
if st.button("Predict"):
    input_data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.write(f"Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
