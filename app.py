import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="AI Tính Tiền Cơm",
    page_icon="🍱",
    layout="wide"
)

# ==================== DANH SÁCH MÓN ĂN ====================
MENU = {
    "Cơm trắng": {"price": 10000, "note": "Một giá tiền nhiều hay ít"},
    "Đậu hũ sốt cà": {"price": 25000, "note": ""},
    "Cá hú kho": {"price": 30000, "note": ""},
    "Thịt kho trứng": {"price": 30000, "note": "Một trứng, thêm 1 trứng + 6.000đ"},
    "Thịt kho": {"price": 25000, "note": "Không có trứng"},
    "Canh chua có cá": {"price": 25000, "note": ""},
    "Canh chua không cá": {"price": 10000, "note": ""},
    "Sườn nướng": {"price": 30000, "note": ""},
    "Canh rau": {"price": 7000, "note": "Cải hay muống"},
    "Rau xào": {"price": 10000, "note": "Lagim/củ sắn/đậu que/đậu đũa"},
    "Trứng chiên": {"price": 25000, "note": "Trứng chiên thịt"}
}

# ==================== HÀM CROP ẢNH ====================
def crop_food_tray(image):
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (1400, 1300))
    
    regions = {
        "vung_1": img[40:700, 40:760],
        "vung_2": img[40:700, 820:1380],
        "vung_3": img[760:1280, 30:500],
        "vung_4": img[760:1280, 520:920],
        "vung_5": img[760:1280, 950:1380]
    }
    
    return regions

# ==================== HÀM DỰ ĐOÁN ====================
def predict_food(model, image):
    regions = crop_food_tray(image)
    results = []
    
    for name, crop in regions.items():
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        img_array = np.expand_dims(crop_resized, axis=0) / 255.0
        
        pred = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(pred)
        confidence = np.max(pred)
        
        # Lấy tên món ăn từ class_names (sẽ được set sau khi load model)
        results.append({
            "region": name,
            "food": class_names[pred_idx] if 'class_names' in globals() else f"Mon_{pred_idx}",
            "confidence": confidence,
            "image": crop_rgb
        })
    
    return results

# ==================== GIAO DIỆN ====================
st.title("🍱 AI Tính Tiền Cơm")
st.markdown("---")

# Sidebar - Chọn chế độ
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    mode = st.radio(
        "Chọn chế độ",
        ["📊 Tính tiền thủ công", "🤖 Tính tiền tự động (AI)"]
    )
    
    st.markdown("---")
    st.header("📋 Danh sách món ăn")
    
    for food, info in MENU.items():
        st.write(f"**{food}**: {info['price']:,} VND")
        if info['note']:
            st.caption(f"  📝 {info['note']}")

# ==================== MODE 1: TÍNH TIỀN THỦ CÔNG ====================
if mode == "📊 Tính tiền thủ công":
    st.header("📊 Nhập món ăn")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        order = []
        selected_foods = st.multiselect(
            "Chọn món ăn:",
            list(MENU.keys())
        )
        
        for food in selected_foods:
            qty = st.number_input(
                f"Số lượng {food}:",
                min_value=1,
                max_value=10,
                value=1,
                key=f"qty_{food}"
            )
            order.append({"food": food, "qty": qty})
    
    with col2:
        st.subheader("💰 Hóa đơn")
        if order:
            total = 0
            for item in order:
                price = MENU[item["food"]]["price"] * item["qty"]
                total += price
                st.write(f"{item['food']} x{item['qty']}: {price:,} VND")
            
            st.markdown("---")
            st.subheader(f"Tổng cộng: {total:,} VND")
        else:
            st.write("Chưa có món ăn nào")

# ==================== MODE 2: AI TỰ ĐỘNG ====================
else:
    st.header("🤖 Tính tiền tự động với AI")
    
    # Kiểm tra model đã load chưa
    if 'model' not in st.session_state:
        st.warning("⚠️ Vui lòng tải model AI trước khi sử dụng!")
        st.info("Upload file model .keras hoặc .tflite để bắt đầu")
        
        uploaded_model = st.file_uploader(
            "Tải model lên:",
            type=['keras', 'h5', 'tflite']
        )
        
        if uploaded_model is not None:
            with st.spinner("Đang tải model..."):
                # Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                    tmp.write(uploaded_model.getvalue())
                    model_path = tmp.name
                
                try:
                    from tensorflow.keras.models import load_model
                    st.session_state.model = load_model(model_path)
                    st.session_state.class_names = [
                        'Cơm trắng', 'Đậu hũ sốt cà', 'Cá hú kho', 
                        'Thịt kho trứng', 'Thịt kho', 'Canh chua có cá',
                        'Canh chua không cá', 'Sườn nướng', 'Canh rau', 
                        'Rau xào', 'Trứng chiên'
                    ]
                    st.success("✅ Đã tải model thành công!")
                    os.unlink(model_path)
                except Exception as e:
                    st.error(f"❌ Lỗi tải model: {str(e)}")
    else:
        # Đã có model
        st.success("✅ Model AI đã sẵn sàng!")
        
        # Upload ảnh
        uploaded_file = st.file_uploader(
            "📤 Tải ảnh khay cơm lên:",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Đọc ảnh
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh khay cơm", use_container_width=True)
            
            if st.button("🔍 Nhận diện và tính tiền"):
                with st.spinner("Đang xử lý..."):
                    # Dự đoán
                    model = st.session_state.model
                    global class_names
                    class_names = st.session_state.class_names
                    
                    results = predict_food(model, image)
                    
                    # Hiển thị kết quả
                    st.subheader("📊 Kết quả nhận diện")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Hiển thị các vùng đã crop
                        cols = st.columns(3)
                        for idx, result in enumerate(results):
                            with cols[idx % 3]:
                                st.image(result['image'], caption=f"Vùng {idx+1}", use_container_width=True)
                                st.write(f"**{result['food']}**")
                                st.write(f"Độ tin cậy: {result['confidence']:.2%}")
                    
                    with col2:
                        # Tính tiền
                        st.subheader("💰 Hóa đơn")
                        total = 0
                        for result in results:
                            food_name = result['food']
                            if food_name in MENU:
                                price = MENU[food_name]['price']
                                total += price
                                st.write(f"{food_name}: {price:,} VND")
                            else:
                                st.write(f"{food_name}: Không có giá")
                        
                        st.markdown("---")
                        st.subheader(f"Tổng cộng: {total:,} VND")
                        
                        # Nút reset
                        if st.button("🔄 Reset"):
                            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.caption("© 2024 AI Tính Tiền Cơm - Đồ án AI CUỐI KỲ")
