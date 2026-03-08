import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap


st.set_page_config(page_title="Hydraulic System Monitoring", layout="wide")


@st.cache_data
def load_data():

    data_path = 'data/' 
    sensor_files = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE']
    

    feature_df = pd.DataFrame()
    for name in sensor_files:
        
        df = pd.read_csv(f"{data_path}{name}.txt", sep='\t', header=None)
        feature_df[f'{name}_mean'] = df.mean(axis=1)
        feature_df[f'{name}_std'] = df.std(axis=1)
        feature_df[f'{name}_max'] = df.max(axis=1)
        feature_df[f'{name}_min'] = df.min(axis=1)
        
    labels = pd.read_csv(f"{data_path}profile.txt", sep='\t', header=None)
    labels.columns = ['Cooler_Condition', 'Valve_Condition', 'Internal_Pump_Leakage', 'Hydraulic_Accumulator', 'Stable_Flag']
    
    return feature_df, labels


st.sidebar.title("🛠 Dashboard Control")
page = st.sidebar.selectbox("Choose Page", ["Data Explorer", "Failure Diagnosis", "Model Insights"])

try:
    df_features, df_labels = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure data files are in the 'data/' folder.")
    st.stop()

# --- الصفحة الأولى: استكشاف البيانات ---
if page == "Data Explorer":
    st.title("📊 Hydraulic System Data Explorer")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Features Preview")
        st.dataframe(df_features.head(10))
    
    with col2:
        st.subheader("Target Distributions")
        target = st.selectbox("Select Component to Visualize", df_labels.columns)
        fig, ax = plt.subplots()
        sns.countplot(x=df_labels[target], ax=ax, palette="viridis")
        st.pyplot(fig)

    st.divider()
    st.subheader("Sensor Correlation Heatmap")
    # عرض مصفوفة الارتباط لأول 10 حساسات فقط للوضوح
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_features.iloc[:, :15].corr(), annot=False, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# --- الصفحة الثانية: التنبؤ والتشخيص ---
elif page == "Failure Diagnosis":
    st.title("🔍 Failure Diagnosis & Prediction")
    
    st.write("This section uses trained Random Forest models to predict the health of hydraulic components.")
    
    # اختيار الحالة المراد التنبؤ بها
    target_to_predict = st.selectbox("Select Component to Diagnose", 
                                    ['Cooler_Condition', 'Valve_Condition', 'Internal_Pump_Leakage', 'Hydraulic_Accumulator'])
    
    # تدريب الموديل (بشكل سريع للعرض)
    y = df_labels[target_to_predict]
    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # اختيار عينة عشوائية للتجربة
    sample_idx = st.slider("Select Sample Row for Diagnosis", 0, len(X_test)-1, 10)
    sample_data = X_test.iloc[[sample_idx]]
    
    prediction = model.predict(sample_data)[0]
    
    # عرض النتيجة 
    st.markdown("---")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Predicted State", value=str(prediction))
        st.write(f"Actual State: **{y_test.iloc[sample_idx]}**")
        
        if prediction == y_test.iloc[sample_idx]:
            st.success("✅ Prediction Matches Actual State")
        else:
            st.warning("⚠️ Prediction Mismatch")

    with res_col2:
        st.subheader("Sensor Readings for this Sample")
        st.bar_chart(sample_data.T.iloc[:10]) # عرض أول 10 قيم

# --- الصفحة الثالثة: شرح الموديل ---
elif page == "Model Insights":
    st.title("🧠 Model Interpretability (SHAP)")
    
    target_to_explain = st.selectbox("Explain Model For:", 
                                    ['Cooler_Condition', 'Valve_Condition', 'Internal_Pump_Leakage', 'Hydraulic_Accumulator'])
    
    # تدريب موديل سريع
    y = df_labels[target_to_explain]
    X_train, X_test, y_train, y_test = train_test_split(df_features, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(X_train, y_train)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance (Built-in)")
        importances = pd.Series(model.feature_importances_, index=df_features.columns).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots()
        importances.plot(kind='barh', ax=ax, color='skyblue')
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    with col2:
        st.subheader("SHAP Summary Plot")
        explainer = shap.TreeExplainer(model)
        # نأخذ عينة صغيرة للـ SHAP لأنه يستهلك وقتاً
        shap_values = explainer.shap_values(X_test.iloc[:50], check_additivity=False)
        
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, X_test.iloc[:50], plot_type="bar", show=False)
        st.pyplot(plt.gcf())

# تذييل الصفحة
st.sidebar.markdown("---")
st.sidebar.info("Developed by: Mohamedien Khalfalla | AI Condition Monitoring")