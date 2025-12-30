import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ==================== CSS TÙY CHỈNH ====================
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9068 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.6s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .bad { 
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #e57373;
    }
    .good { 
        background-color: #e8f5e9;
        color: #2e7d32; 
        border: 2px solid #81c784;
    }
    h1, h2, h3 {
        color: #0e1117;
        font-weight: 700;
    }
    /* Container styling with forced text color for dark mode compatibility */
    .feature-group {
        background-color: #ffffff;
        color: #31333F; /* Dark text for white background */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 25px;
    }
    .feature-group h5 {
        color: #31333F !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 0.5rem;
    }
    /* Fix input labels inside the white container if needed */
    /* Fix input labels inside the white container if needed */
    .stNumberInput label, .stNumberInput label p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODELS ====================
base_dir = Path(__file__).parent
models_dir = base_dir / "models"

@st.cache_resource
def load_models():
    try:
        knn = joblib.load(models_dir / "knn.pkl")
        svm = joblib.load(models_dir / "svm.pkl")
        rf = joblib.load(models_dir / "rf.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        return knn, svm, rf, scaler, None
    except Exception as e:
        return None, None, None, None, str(e)

knn, svm, rf, scaler, error = load_models()

if error:
    st.error(f"❌ Lỗi không thể tải models: {error}")
    st.info("Vui lòng đảm bảo bạn đã chạy notebook để train và save models vào thư mục 'models/'.")
    st.stop()

# ==================== GIAO DIỆN ====================
st.title("🍷 Dự đoán Chất lượng Rượu Vang")
st.markdown("### Ứng dụng AI phân loại chất lượng rượu (Good/Bad)")
st.markdown("---")

# Sidebar - Chọn Model
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80", use_container_width=True)
    st.header("⚙️ Cấu hình Mô hình")
    
    model_options = ["Random Forest", "KNN", "SVM"]
    selected_model_name = st.selectbox(
        "Chọn thuật toán:",
        model_options,
        index=0
    )
    
    if selected_model_name == "Random Forest":
        st.success("✅ **Random Forest**: Độ chính xác cao nhất, ổn định.")
    elif selected_model_name == "KNN":
        st.info("ℹ️ **KNN**: Dựa trên độ tương đồng của láng giềng.")
    else:
        st.warning("⚠️ **SVM**: Tốt cho không gian nhiều chiều.")
    
    st.markdown("---")
    st.info("""
    **Thông tin phân loại:**
    - **Good (Tốt)**: Chất lượng >= 7
    - **Bad (Chưa tốt)**: Chất lượng < 7
    """)

    st.markdown("---")
    st.header("⚡ Dữ liệu Mẫu (Demo)")
    
    def update_param(key, value):
        st.session_state[key] = value
        st.session_state[f"{key}_input"] = value
        st.session_state[f"{key}_slider"] = value

    c_demo1, c_demo2 = st.columns(2)
    with c_demo1:
        if st.button("🍷 Mẫu Tốt"):
            import random
            update_param("fixed_acidity", round(random.uniform(7.0, 10.0), 1))
            update_param("volatile_acidity", round(random.uniform(0.2, 0.45), 2))
            update_param("citric_acid", round(random.uniform(0.3, 0.6), 2))
            update_param("residual_sugar", round(random.uniform(1.5, 4.0), 1))
            update_param("chlorides", round(random.uniform(0.04, 0.08), 3))
            update_param("free_sulfur_dioxide", float(random.randint(15, 30)))
            update_param("total_sulfur_dioxide", float(random.randint(20, 50)))
            update_param("density", round(random.uniform(0.9940, 0.9970), 4))
            update_param("pH", round(random.uniform(3.1, 3.4), 2))
            update_param("sulphates", round(random.uniform(0.65, 0.95), 2))
            update_param("alcohol", round(random.uniform(11.0, 13.5), 1))
            st.rerun()
            
    with c_demo2:
        if st.button("🍇 Mẫu Kém"):
            import random
            update_param("fixed_acidity", round(random.uniform(6.5, 9.0), 1))
            update_param("volatile_acidity", round(random.uniform(0.6, 1.0), 2))
            update_param("citric_acid", round(random.uniform(0.0, 0.25), 2))
            update_param("residual_sugar", round(random.uniform(1.5, 4.5), 1))
            update_param("chlorides", round(random.uniform(0.08, 0.12), 3))
            update_param("free_sulfur_dioxide", float(random.randint(5, 18)))
            update_param("total_sulfur_dioxide", float(random.randint(15, 45)))
            update_param("density", round(random.uniform(0.9970, 1.0000), 4))
            update_param("pH", round(random.uniform(3.4, 3.7), 2))
            update_param("sulphates", round(random.uniform(0.35, 0.60), 2))
            update_param("alcohol", round(random.uniform(9.0, 10.5), 1))
            st.rerun()

    st.markdown("Made with ❤️ by AI Assistant")

def smart_parameter(label, min_v, max_v, default_v, step, key, help_text, fmt="%.2f"):
    if key not in st.session_state:
        st.session_state[key] = default_v
        
    def update_from_input():
        st.session_state[key] = st.session_state[f"{key}_input"]
        st.session_state[f"{key}_slider"] = st.session_state[key]
        
    def update_from_slider():
        st.session_state[key] = st.session_state[f"{key}_slider"]
        st.session_state[f"{key}_input"] = st.session_state[key]

    # Label styling
    st.markdown(f'<p style="margin-bottom: 0px; font-weight: 700; color: #ffffff; font-size: 14px;">{label}</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([35, 65])
    with col1:
        st.number_input(
            label, min_value=float(min_v), max_value=float(max_v), step=float(step), 
            key=f"{key}_input", value=float(st.session_state[key]), 
            on_change=update_from_input, label_visibility="collapsed", format=fmt
        )
    with col2:
        st.slider(
            label, min_value=float(min_v), max_value=float(max_v), step=float(step), 
            key=f"{key}_slider", value=float(st.session_state[key]), 
            on_change=update_from_slider, label_visibility="collapsed", help=help_text
        )
    return st.session_state[key]

# Main - Nhập liệu
st.subheader("📝 Nhập thông số hóa lý của rượu")

# Nhóm 1: Thành phần Axit & Đường
st.markdown('<div class="feature-group"><h5>🍋 Thành phần Axit & Đường</h5>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    fixed_acidity = smart_parameter("Fixed Acidity", 4.0, 16.0, 7.4, 0.1, "fixed_acidity", "Độ chua cố định (g/L)")
    citric_acid = smart_parameter("Citric Acid", 0.0, 1.0, 0.0, 0.01, "citric_acid", "Axit citric (g/L)")
with c2:
    volatile_acidity = smart_parameter("Volatile Acidity", 0.1, 2.0, 0.7, 0.01, "volatile_acidity", "Độ chua bay hơi (g/L)")
    residual_sugar = smart_parameter("Residual Sugar", 0.0, 16.0, 1.9, 0.1, "residual_sugar", "Đường dư (g/L)")
st.markdown('</div>', unsafe_allow_html=True)

# Nhóm 2: Khoáng chất & Sulfur
st.markdown('<div class="feature-group"><h5>🧪 Khoáng chất & Sulfur</h5>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    chlorides = smart_parameter("Chlorides", 0.01, 0.6, 0.076, 0.001, "chlorides", "Muối clorua (g/L)", fmt="%.3f")
    total_sulfur_dioxide = smart_parameter("Total SO2", 6.0, 289.0, 34.0, 1.0, "total_sulfur_dioxide", "Tổng sulfur dioxide (mg/L)", fmt="%.0f")
with c4:
    free_sulfur_dioxide = smart_parameter("Free SO2", 1.0, 72.0, 11.0, 1.0, "free_sulfur_dioxide", "Sulfur dioxide tự do (mg/L)", fmt="%.0f")
    sulphates = smart_parameter("Sulphates", 0.3, 2.0, 0.56, 0.01, "sulphates", "Sunfat (g/Lpotassium sulphate)")
st.markdown('</div>', unsafe_allow_html=True)

# Nhóm 3: Các chỉ số vật lý khác
st.markdown('<div class="feature-group"><h5>📊 Các chỉ số vật lý khác</h5>', unsafe_allow_html=True)
c5, c6, c7 = st.columns(3)
with c5:
    density = smart_parameter("Density", 0.9900, 1.0050, 0.9978, 0.0001, "density", "Tỷ trọng (g/cm³)", fmt="%.4f")
with c6:
    pH = smart_parameter("pH", 2.0, 5.0, 3.51, 0.01, "pH", "Độ pH")
with c7:
    alcohol = smart_parameter("Alcohol", 8.0, 15.0, 9.4, 0.1, "alcohol", "Nồng độ cồn (% vol)")
st.markdown('</div>', unsafe_allow_html=True)

# Submit Button
submitted = st.button("Dự đoán", type="primary")

# ==================== XỬ LÝ DỰ ĐOÁN ====================
if submitted:
    # 1. Thu thập dữ liệu input theo đúng thứ tự lúc train (11 features)
    # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    #  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    #  'pH', 'sulphates', 'alcohol']
    input_features = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol
    ]])

    # 2. Xử lý pre-processing
    try:
        # Scale dữ liệu (dùng cho KNN và SVM)
        input_scaled = scaler.transform(input_features)
        
        # Chọn model và input phù hợp
        if selected_model_name == "KNN":
            model = knn
        elif selected_model_name == "SVM":
            model = svm
        else:
            model = rf
            
        # Tất cả model (kể cả Random Forest) đều dùng dữ liệu scaled
        # Vì trong notebook, RF được train trên X_train_resampled (là dữ liệu đã scale + SMOTE)
        final_input = input_scaled

        # 3. Dự đoán
        prediction = model.predict(final_input)[0]
        
        # Mapping kết quả (0: Bad, 1: Good)
        if prediction == 1:
            label_vi = "Tốt (Good)"
            label_desc = "Rượu có chất lượng cao, hương vị cân bằng."
            css_class = "good"
            icon = "🥂"
        else:
            label_vi = "Chưa tốt (Bad)"
            label_desc = "Rượu cần cải thiện về chất lượng."
            css_class = "bad"
            icon = "🍇"

        # 4. Hiển thị kết quả
        st.markdown(f"""
            <div class="result-card {css_class}">
                <h3>KẾT QUẢ PHÂN TÍCH</h3>
                <div style="font-size: 60px;">{icon}</div>
                <h1 style="margin: 10px 0;">{label_vi.upper()}</h1>
                <p style="font-size: 18px;">{label_desc}</p>
            </div>
        """, unsafe_allow_html=True)

        # Hiển thị độ tin cậy (Probability)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(final_input)[0]
            confidence = probs[prediction] * 100
            
            st.markdown("#### 🎯 Độ tin cậy của mô hình:")
            st.progress(int(confidence))
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Xác suất là Rượu Tốt", f"{probs[1]*100:.1f}%")
            with col_p2:
                st.metric("Xác suất là Rượu Chưa Tốt", f"{probs[0]*100:.1f}%")

    except Exception as e:
        st.error(f"⚠️ Đã xảy ra lỗi trong quá trình dự đoán: {str(e)}")
