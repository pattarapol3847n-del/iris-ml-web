import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= ข้อมูลผู้พัฒนา =================
st.sidebar.markdown("---")
st.sidebar.image("my_photo.jpg", width=150)
st.sidebar.info("""
**ผู้พัฒนา:** ภัทรพล แก้วแท้
**รหัสนักศึกษา:** 664245029
**หมู่เรียน:** 66/43
""")
st.sidebar.markdown("---")
# ================================================

st.set_page_config(page_title="Iris ML Predictor", page_icon="🌸")
st.title("🌸 ทำนายสายพันธุ์ดอกไม้ Iris")

@st.cache_resource
def load_models():
    m = {
        'K-Nearest Neighbor': joblib.load('model_K_Nearest_Neighbor.pkl'),
        'Decision Tree': joblib.load('model_Decision_Tree.pkl'),
        'SVM': joblib.load('model_SVM.pkl'),
        'Logistic Regression': joblib.load('model_Logistic_Regression.pkl'),
        'Random Forest': joblib.load('model_Random_Forest.pkl'),
        'K-Means': joblib.load('model_summary.pkl') # โหลดไว้ก่อน เผื่อใช้ได้
    }
    return m, joblib.load('scaler.pkl')

try:
    models, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

selected = st.sidebar.selectbox("เลือกโมเดล:", list(models.keys()))

c1, c2 = st.columns(2)
with c1:
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.5, 0.1)
    sw = st.slider("Sepal Width", 2.0, 4.5, 3.0, 0.1)
with c2:
    pl = st.slider("Petal Length", 1.0, 7.0, 4.0, 0.1)
    pw = st.slider("Petal Width", 0.1, 2.6, 1.5, 0.1)

if st.button(" ทำนายผล", type="primary"):
    inp = np.array([[sl, sw, pl, pw]])
    
    # --- ส่วนที่แก้ไข: เพิ่ม Try-Except ดัก Error ---
    try:
        # ลองใช้ Scaler ก่อนสำหรับ SVM/Logistic
        if selected in ['SVM', 'Logistic Regression']:
            inp_s = scaler.transform(inp)
            pred = models[selected].predict(inp_s)
        # สำหรับ K-Means ลองใช้ข้อมูลดิบ (inp) ก่อน
        elif selected == 'K-Means':
            pred = models[selected].predict(inp) 
        else:
            # โมเดลอื่นๆ ใช้ข้อมูลดิบ
            pred = models[selected].predict(inp)
            
        res = pred[0]
        
        # แปลงผลลัพธ์ K-Means
        if selected == 'K-Means':
            cmap = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
            res = cmap.get(int(pred[0]), 'Unknown')
            
        st.success(f"ผลลัพธ์: **{res}**")
        
        # แสดงกราฟ (ยกเว้น K-Means)
        if hasattr(models[selected], 'predict_proba') and selected != 'K-Means':
            data_to_predict = inp_s if selected in ['SVM', 'Logistic Regression'] else inp
            prob = models[selected].predict_proba(data_to_predict)
            pdf = pd.DataFrame(prob, columns=['Setosa', 'Versicolor', 'Virginica']).T
            st.bar_chart(pdf)

    except Exception as e:
        # ถ้า K-Means มีปัญหา จะขึ้นข้อความนี้แทนที่จะพังทั้งเว็บ
        if selected == 'K-Means':
            st.warning("⚠️ โมเดล K-Means กำลังอยู่ในระหว่างปรับปรุง (Model File Issue) กรุณาเลือกโมเดลอื่นเพื่อทดสอบครับ")
            st.caption(f"Technical Error: {str(e)[:100]}...")
        else:
            st.error(f"เกิดข้อผิดพลาด: {e}")
