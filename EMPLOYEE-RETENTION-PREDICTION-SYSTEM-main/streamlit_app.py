import streamlit as st
import pickle
import numpy as np

# Load the model
with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the main function for the Streamlit app
def main():
    st.title('Employee Churn Prediction')
    
    # Input fields
    satisfaction_level = st.number_input('Satisfaction Level', min_value=0.0, max_value=1.0, step=0.01)
    last_evaluation = st.number_input('Last Evaluation', min_value=0.0, max_value=1.0, step=0.01)
    number_project = st.number_input('Number of Projects', min_value=1, max_value=10, step=1)
    average_montly_hours = st.number_input('Average Monthly Hours', min_value=0, max_value=300, step=1)
    time_spend_company = st.number_input('Time Spent at Company', min_value=1, max_value=20, step=1)
    Work_accident = st.selectbox('Work Accident', [0, 1])
    promotion_last_5years = st.selectbox('Promotion in Last 5 Years', [0, 1])
    low_salary = st.selectbox('Low Salary', [0, 1])
    medium_salary = st.selectbox('Medium Salary', [0, 1])

    # Prepare input data
    input_data = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours,
                            time_spend_company, Work_accident, promotion_last_5years, low_salary, medium_salary]])
    
    # Predict and display result
    if st.button('Predict'):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.write('The employee is likely to leave.')
        else:
            st.write('The employee is likely to stay.')

if __name__ == '__main__':
    main()
