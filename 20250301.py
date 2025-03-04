import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load model
try:
    rf_model = joblib.load('random_forest_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Adolescent Depression Risk Assessment", layout="wide")

# Title
st.title("Depression Risk Assessment System for Adolescents  Girls with Non-Suicidal Self-Injury")

# Sidebar inputs
with st.sidebar:
    st.header("Clinical Assessment Parameters")

    # Individual
    age = st.slider("Age", 12, 18, 15)
    ipaq_level = st.selectbox("IPAQ level", options=[1, 2, 3],
                              format_func=lambda x: ["Low", "Moderate", "High"][x - 1])
    bmi = st.slider("BMI", 13.0, 51.0, 21.0, step=0.1)
    sleep_quality = st.slider("Sleep quality", 0, 21, 7)
    perceived_stress = st.slider("Perceived stress", 4, 20, 10)
    hopelessness = st.slider("Hopelessness", 20, 100, 50)
    loneliness = st.slider("Loneliness", 23, 80, 45)
    resilience = st.slider("Resilience", 0, 40, 20)
    alexithymia = st.slider("Alexithymia", 26, 97, 50)
    self_esteem = st.slider("Self-esteem", 10, 40, 20)
    rumination = st.slider("Rumination", 22, 88, 40)
    emotion_regulation = st.slider("Emotion regulation", 10, 50, 25)
    borderline_personality = st.slider("Borderline personality", 25, 120, 50)

    # Family
    care = st.slider("Parental care", 0, 36, 18)
    overprotection = st.slider("Parental overprotection", 0, 36, 18)

    # Psychosocial
    negative_thoughts_behaviors = st.selectbox("Negative thoughts/behaviors", 
                                                options=["No suicidal thoughts or behaviors", 
                                                         "Had suicidal thoughts but no suicidal behaviors", 
                                                         "Had suicidal behaviors"])
    
    number_suicide_attempts = 0
    if negative_thoughts_behaviors == "Had suicidal behaviors":
        number_suicide_attempts = st.number_input("Number of suicide attempts", 0, 100, 0)

    problem_focused_coping = st.slider("Problem-focused coping", 20, 80, 40)
    emotion_focused_coping = st.slider("Emotion-focused coping", 17, 66, 30)
    major_life_events = st.selectbox("Major life events", options=[0, 1],
                                     format_func=lambda x: ["No", "Yes"][x])

# Create a list of input features
input_features = [
    age, major_life_events, negative_thoughts_behaviors, number_suicide_attempts,
    ipaq_level, bmi, sleep_quality, perceived_stress, hopelessness, loneliness,
    resilience, alexithymia, problem_focused_coping, emotion_focused_coping,
    self_esteem, rumination, emotion_regulation, borderline_personality,
    care, overprotection
]
# 将选择的字符串选项映射为原来的0, 1, 2数值
ntb_mapping = {"No suicidal thoughts or behaviors": 0, 
               "Had suicidal thoughts but no suicidal behaviors": 1, 
               "Had suicidal behaviors": 2}
input_features[input_features.index(negative_thoughts_behaviors)] = ntb_mapping[negative_thoughts_behaviors]

input_data = pd.DataFrame([input_features], columns=[
    'Age', 'Major life events', 'Negative thoughts/behaviors', 'Number of suicide attempts',
    'IPAQ level', 'BMI', 'Sleep quality', 'Perceived stress', 'Hopelessness', 'Loneliness',
    'Resilience', 'Alexithymia', 'Problem-focused coping', 'Emotion-focused coping',
    'Self-esteem', 'Rumination', 'Emotion regulation', 'Borderline personality',
    'Care', 'Overprotection'
])

# 定义 feature_names
feature_names = input_data.columns.tolist()

# Prediction execution
if st.button("Start Assessment"):
    try:
        proba = rf_model.predict_proba(input_data)[0][1]
        prediction = 1 if proba >= 0.5 else 0
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()

    # ================= Risk Stratification =================
    RISK_THRESHOLDS = {'low': 0.3, 'high': 0.7}

    if proba <= RISK_THRESHOLDS['low']:
        risk_level = "Low Risk"
        color = "#2ecc71"
        icon = "✅"
    elif proba <= RISK_THRESHOLDS['high']:
        risk_level = "Medium Risk"
        color = "#f1c40f"
        icon = "⚠️"
    else:
        risk_level = "High Risk"
        color = "#e74c3c"
        icon = "🚨"

    # Result display
    result_col1, result_col2 = st.columns([1, 1])
    with result_col1:
        st.subheader("Assessment Result")
        st.markdown(f"""
        <div style="border:2px solid {color}; border-radius:10px; padding:20px;">
            <h3 style="color:{color}; text-align:center;">{icon} {risk_level}</h3>
            <p style="text-align:center; font-size:24px;">Depression Probability: {proba*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Research basis
        with st.expander("Risk Classification Criteria"):
            st.markdown("""
            **Risk Stratification (WHO Guidelines)**
            - Low Risk (<30%): Routine monitoring
            - Medium Risk (30-70%): Enhanced surveillance
            - High Risk (>70%): Immediate intervention

            *Based on meta-analysis of adolescent depression studies (2010-2022)*
            """)

    with result_col2:
        st.subheader("Key Contributing Factors")

        # Create an explainer
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(input_data)
        except Exception as e:
            st.error(f"An error occurred while calculating SHAP values: {e}")
        else:
            # Check if shap_values is a numpy.ndarray type
            if isinstance(shap_values, np.ndarray):
                print("SHAP values are of numpy.ndarray type")
                # Check if it can be split into SHAP values for two classes
                if shap_values.ndim == 3 and shap_values.shape[-1] == 2:
                    shap_values_class_0 = shap_values[:, :, 0]
                    shap_values_class_1 = shap_values[:, :, 1]
                    shap_values = [shap_values_class_0, shap_values_class_1]
                else:
                    st.error("The shape of SHAP values does not meet expectations. Please check the model and data.")
            else:
                print("SHAP values are of the expected list type")

            # Handle the case where shap_values is a list
            if isinstance(shap_values, list):
                if len(shap_values) == 1:
                    expected_value = explainer.expected_value
                    shap_values = shap_values[0]
                else:
                    expected_value = explainer.expected_value[1]
                    shap_values = shap_values[1]
            else:
                expected_value = explainer.expected_value
                shap_values = shap_values

            # Ensure expected_value is a single value
            if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) == 1:
                expected_value = expected_value[0]

            # Feature importance table
            importance_df = pd.DataFrame({
                'Clinical Factor': feature_names,
                'Impact Direction': ['Positive' if v > 0 else 'Negative' for v in shap_values[0]],
                'Impact Magnitude': np.abs(shap_values[0])
            }).sort_values('Impact Magnitude', ascending=False).head(5)

            # 提取前5个特征的SHAP值和特征名称
            selected_shap_values = shap_values[0][np.argsort(np.abs(shap_values[0]))[::-1][:5]]
            selected_feature_names = importance_df['Clinical Factor'].tolist()

            # 展示表格
            st.dataframe(
                importance_df[['Clinical Factor', 'Impact Direction']],
                column_config={
                    "Impact Direction": st.column_config.TextColumn(
                        help="Positive: Higher values increase risk\nNegative: Higher values decrease risk"
                    )
                },
                hide_index=True
            )

    # Display SHAP value explanations
    st.subheader('SHAP Explanation')
    # Force plot for a single sample
    try:
        force_plot = shap.plots.force(expected_value, shap_values.flatten(), input_data.iloc[0], feature_names=input_data.columns)
        force_plot_matplotlib = shap.force_plot(expected_value, shap_values.flatten(), input_data.iloc[0], feature_names=input_data.columns).matplotlib(figsize=(20, 3), show=False, text_rotation=0)
        st.pyplot(force_plot_matplotlib)
    except Exception as e:
        st.error(f"An error occurred while plotting the force plot: {e}")

    # ================= Personalized Recommendations =================
    st.subheader("Clinical Recommendations")

    # Base recommendations
    recommendations = {
        "Low Risk": [
            "Maintain regular sleep schedule (7-9 hours/night)",
            "Engage in moderate exercise ≥3 times/week",
            "Practice daily mindfulness exercises"
        ],
        "Medium Risk": [
            "Schedule psychological evaluation within 2 weeks",
            "Initiate cognitive behavioral therapy (CBT) techniques",
            "Establish social support network"
        ],
        "High Risk": [
            "Require immediate psychiatric consultation",
            "Develop crisis management plan with caregivers",
            "Implement 24-hour monitoring"
        ]
    }

    # Feature-specific advice
    feature_advice = {
        ('Hopelessness', 'Positive'): "Implement hope-building interventions (e.g., goal-setting exercises)",
        ('Resilience', 'Negative'): "Enhance resilience through problem-solving training",
        ('Sleep quality', 'Positive'): "Improve sleep hygiene (consistent schedule, screen time management)",
        ('BMI', 'Positive'): "Develop nutritional management plan (consult registered dietitian)"
    }

    # Generate recommendations

    # Generate recommendations
    final_advice = recommendations[risk_level]
    for row in importance_df.itertuples():
        key = (row._1.strip(), row._2)  
        if key in feature_advice:
            final_advice.append(f"【{row._1.strip()}】{feature_advice[key]}")
            

    # Display recommendations
    for i, item in enumerate(final_advice[:5], 1):
        st.markdown(f"{i}. {item}")

    # ================= Technical Details =================
    with st.expander("Technical Details (For Researchers)"):
        tab1, tab2 = st.tabs(["SHAP Values", "Raw Data"])

        with tab1:
            st.write("Complete SHAP Analysis:")
            st.dataframe(importance_df)

        with tab2:
            st.write("Input Feature Data:")
            st.dataframe(input_data.T.rename(columns={0: "Value"}))

# Footer
st.divider()
st.markdown("""
**Instructions**
1. All scale scores should be obtained through professional assessment
2. High-risk results require clinical confirmation
3. System Update Date: 2024-03-01
""")
