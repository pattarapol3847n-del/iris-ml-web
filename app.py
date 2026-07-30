import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= ข้อมูลผู้พัฒนา (เพิ่มตามโจทย์) =================
st.sidebar.markdown("---")
st.sidebar.image("my_photo.jpg", width=150) # แสดงรูปที่อัปโหลดไว้
st.sidebar.info("""
**ผู้พัฒนา:** ภัทรพล แก้วแท้

**รหัสนักศึกษา:** 664245029

**หมู่เรียน:** 66/43
""")
st.sidebar.markdown("---")
# ================================================================

st.set_page_config(page_title="Iris ML Predictor", page_icon="")
st.title("🌸 ทำนายสายพันธุ์ดอกไม้ Iris")

@st.cache_resource
def load_models():
    m = {
        'K-Nearest Neighbor': joblib.load('model_K_Nearest_Neighbor.pkl'),
        'Decision Tree': joblib.load('model_Decision_Tree.pkl'),
        'SVM': joblib.load('model_SVM.pkl'),
        'Logistic Regression': joblib.load('model_Logistic_Regression.pkl'),
        'Random Forest': joblib.load('model_Random_Forest.pkl'),
        'K-Means': joblib.load('model_K_Means.pkl')
    }
    return m, joblib.load('scaler.pkl')

try:
    models, scaler = load_models()
except:
    st.error("Error loading models. Please ensure all .pkl files are present.")
    st.stop()

selected = st.sidebar.selectbox("เลือกโมเดล:", list(models.keys()))

c1, c2 = st.columns(2)
with c1:
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.5, 0.1)
    sw = st.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1)
with c2:
    pl = st.slider("Petal Length", 1.0, 7.0, 4.0, 0.1)
    pw = st.slider("Petal Width", 0.1, 2.6, 1.5, 0.1)

if st.button("🔮 ทำนายผล", type="primary"):
    inp = np.array([[sl, sw, pl, pw]])
    if selected in ['SVM', 'Logistic Regression', 'K-Means']:
        inp_s = scaler.transform(inp)
        pred = models[selected].predict(inp_s)
    else:
        pred = models[selected].predict(inp)
        
    res = pred[0]
    if selected == 'K-Means':
        cmap = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        res = cmap.get(pred[0], 'Unknown')
        
    st.success(f"ผลลัพธ์: **{res}**")
    
    if hasattr(models[selected], 'predict_proba') and selected != 'K-Means':
        prob = models[selected].predict_proba(inp_s if selected in ['SVM', 'Logistic Regression'] else inp)
        pdf = pd.DataFrame(prob, columns=['Setosa', 'Versicolor', 'Virginica']).T
        st.bar_chart(pdf)
