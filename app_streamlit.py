"""
Version Streamlit avec envoi API - Interface Web
"""

import cv2
import pickle
import cvzone
import numpy as np
import streamlit as st
import time
import requests
from datetime import datetime

# ==================== CONFIGURATION ====================
API_URL = st.sidebar.text_input("API URL", "http://172.20.10.3:8081/api/iot/detection")
PARKING_ID = st.sidebar.number_input("Parking ID", min_value=1, value=1)
SEND_TO_API = st.sidebar.checkbox("Envoyer à l'API", value=True)

# Dimensions
PLACE_WIDTH = 108
PLACE_HEIGHT = 48
SEUIL_LIBRE = 900

# Charger les positions
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []
    st.error("Fichier CarParkPos non trouvé! Lancez d'abord parkingspacepicker.py")

def checkParkingSpace(img, imgPro):
    spaceCounter = 0
    detections = []
    
    for i, pos in enumerate(posList):
        x, y = pos
        imgCrop = imgPro[y:y + PLACE_HEIGHT, x:x + PLACE_WIDTH]
        count = cv2.countNonZero(imgCrop)

        if count < SEUIL_LIBRE:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            etat = 0
        else:
            color = (0, 0, 255)
            thickness = 2
            etat = 1

        cv2.rectangle(img, pos, (pos[0] + PLACE_WIDTH, pos[1] + PLACE_HEIGHT), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + PLACE_HEIGHT - 3), scale=1, thickness=2, offset=0, colorR=color)
        
        detections.append({
            "placeId": i + 1,
            "etat": etat,
            "confidence": 0.95
        })
    
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0,200,0))
    return img, detections, spaceCounter

# Interface Streamlit
st.set_page_config(page_title="ParkiHna", layout="wide")
st.title("🚗 Smart Car Parking System")

# Sidebar
st.sidebar.header("Settings")
fps_limit = st.sidebar.slider("Frame Rate Limit (FPS)", 1, 30, 2)

video_source = st.sidebar.radio("Select Video Source", ("Default Video", "Upload Your Own"))

if video_source == "Default Video":
    cap = cv2.VideoCapture('carPark.mp4')
else:
    uploaded_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi"])
    if uploaded_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture("temp_video.mp4")
    else:
        cap = None

# Dernier état envoyé
last_sent_state = {}
last_send_time = time.time()

if cap and posList:
    st_frame = st.empty()
    
    while True:
        start_time = time.time()
        
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Prétraitement
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        # Traitement
        img, detections, spaceCounter = checkParkingSpace(img, imgDilate)

        # Envoi à l'API
        if SEND_TO_API and time.time() - last_send_time > 2:
            changes = []
            for det in detections:
                place_id = det["placeId"]
                if last_sent_state.get(place_id) != det["etat"]:
                    changes.append(det)
                    last_sent_state[place_id] = det["etat"]
            
            if changes:
                payload = {
                    "parkingId": PARKING_ID,
                    "ipRaspberry": "streamlit-client",
                    "idCamera": "STREAMLIT-01",
                    "detections": changes
                }
                try:
                    response = requests.post(API_URL, json=payload, timeout=5)
                    if response.status_code == 200:
                        st.sidebar.success(f"✓ Envoyé: {len(changes)} changements")
                    else:
                        st.sidebar.error(f"✗ Erreur: {response.status_code}")
                except:
                    st.sidebar.error("✗ API inaccessible")
            last_send_time = time.time()

        # Affichage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st_frame.image(img_rgb, channels="RGB")
        
        time.sleep(1 / fps_limit)
