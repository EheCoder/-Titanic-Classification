import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("titanic_survival_model.joblib")

# Function to predict survival
def predict_survival(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S','FamilySize']]
    prediction = model.predict(input_df)
    return prediction[0]

# Main Streamlit app
def main():
    st.title("Titanic Survival Predictor")
    st.sidebar.header("Enter Passenger Details")
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
    sex = st.sidebar.radio("Sex", options=["male", "female"])
    pclass = st.sidebar.selectbox("Passenger Class", options=[1, 2, 3], index=2)
    sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, value=0)
    parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, value=0)
    fare = st.sidebar.number_input("Fare", min_value=0.0, value=10.0)
    embarked = st.sidebar.selectbox("Embarked", options=["C", "Q", "S"], index=2)

    # Prepare input data
    input_data = {
        "Age": age,
        "Sex": 0 if sex == "male" else 1,
        "Pclass": pclass,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_C": True if embarked == "C" else False,
        "Embarked_Q": True if embarked == "Q" else True,
        "Embarked_S": True if embarked == "S" else True,
        "FamilySize": sibsp + parch
    }

    # Predict survival
    prediction = predict_survival(input_data)
    if prediction == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")

if __name__ == "__main__":
    main()
