# 分別利用KNN/ SVM/ Mediapipe三種模型進行人臉偵測
# 整體流程包含：圖片上傳 --＞格式解碼 --＞ KNN 模型辨識人臉 --＞ 繪製標籤與外框 --＞ Streamlit 視覺化與自動存檔。
import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image
from datetime import datetime

# 載入SVM函式庫
import  face_recognition
from  sklearn import  svm



st.title("KNN/ SVM/ Mediapipe三種模型進行人臉偵測")
save_folder = "KNN_SVM_Mediapipe_face_recognition_saved"
os.makedirs(save_folder, exist_ok=True) # os.makedirs() 函數用於創建多層目錄
uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])
# 將所有圖像處理邏輯，放在 if 裡面
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgBGR = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # 解碼成 NumPy 陣列 (BGR)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # 轉換成 RGB 格式 
    original_img = imgRGB.copy() # 複製給 original_img，位於記憶體中的 RGB 格式 NumPy 陣列
    
    
    #原始上傳圖像
    st.subheader("原始上傳圖像")
    st.image(original_img, use_container_width=True) # st.image() 預設接受 RGB

    
    
    # ----SVM人臉偵測(支援向量機無信心指數)---
        
    SVM_bgr = imgBGR.copy()
    SVM_rgb = original_img.copy() # 直接使用已解碼好的陣列
    # 晝框
    # SVM 運算點face_recognition.face_locations，這行的底層運算邏輯正是 HOG (方向梯度直方圖) + 線性 SVM (Support Vector Machine)
    face_locations = face_recognition.face_locations(SVM_rgb) # 輸出 [(120, 340, 280, 180)] 座標資料，
    for (top, right, bottom, left) in face_locations:
        # OpenCV 畫框 (左上點 (left, top)，右下點 (right, bottom))
        cv2.rectangle(SVM_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(SVM_bgr, "Face", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        

    # *將畫好框的 bgr 轉回 RGB，供 Streamlit 正確渲染顏色
    results_rgb = cv2.cvtColor(SVM_bgr, cv2.COLOR_BGR2RGB)   
    # SVM人臉偵測
    st.subheader("SVM人臉偵測")
    st.image(results_rgb, use_container_width=True) # st.image() 預設接受 RGB
    # SVM自動儲存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SVM_filename = os.path.join(save_folder, f"SVM_{timestamp}.png")
    cv2.imwrite(SVM_filename, SVM_bgr)
    st.success(f"SVM人臉偵測已經儲存 {SVM_filename}")

    
   
   

    
 