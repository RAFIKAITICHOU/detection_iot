import cv2
import pickle
import cvzone
import numpy as np
import streamlit as st
import time
import requests
from datetime import datetime

# ==================== CONFIGURATION BACKEND ====================
BACKEND_URL = "http://localhost:8080/api/iot/detection"
PARKING_ID = 1
IP_RASPBERRY = "127.0.0.1"   # Sur PC local, sinon mettre l'IP du Raspberry
ID_CAMERA = "CAM-01"
INTERVALLE_ENVOI = 3          # Secondes entre chaque envoi au backend

# ==================== CHARGEMENT POSITIONS ====================
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# ==================== ENVOI BACKEND ====================
def envoyer_au_backend(etats_places):
    """Envoie les états des places au backend Spring Boot"""
    detections = []
    for i, etat in enumerate(etats_places):
        detections.append({
            "placeId": i + 1,   # placeId commence à 1
            "etat": etat,        # 0=LIBRE, 1=OCCUPEE
            "confidence": 0.95
        })

    payload = {
        "parkingId": PARKING_ID,
        "ipRaspberry": IP_RASPBERRY,
        "idCamera": ID_CAMERA,
        "detections": detections
    }

    try:
        response = requests.post(
            BACKEND_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return True, f"✅ Backend sync OK — {len(detections)} places"
        else:
            return False, f"❌ Erreur backend: HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"⚠️ Backend hors ligne ({BACKEND_URL})"
    except Exception as e:
        return False, f"⚠️ Erreur: {str(e)}"

# ==================== DÉTECTION PLACES ====================
def checkParkingSpace(img, imgPro):
    spaceCounter = 0
    etats = []
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y + 48, x:x + 108]
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            etats.append(0)   # LIBRE
        else:
            color = (0, 0, 255)
            thickness = 2
            etats.append(1)   # OCCUPEE

        cv2.rectangle(img, pos, (pos[0] + 108, pos[1] + 48), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + 48 - 3), scale=1, thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))
    return img, etats

# ==================== INTERFACE STREAMLIT ====================
st.set_page_config(page_title="Smart Parking", layout="wide")
st.title("🚗 Smart Car Parking System")

# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
fps_limit = st.sidebar.slider("Frame Rate Limit (FPS)", 1, 10, 2)

st.sidebar.markdown("---")
st.sidebar.header("🔗 Backend Integration")
backend_enabled = st.sidebar.checkbox("Envoyer au backend", value=True)
intervalle = st.sidebar.slider("Intervalle envoi (sec)", 1, 30, INTERVALLE_ENVOI)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{BACKEND_URL}`")
st.sidebar.markdown(f"**Parking ID:** `{PARKING_ID}`")

# Source vidéo
video_source = st.sidebar.radio("Video Source", ("Default Video", "Upload Your Own"))

if video_source == "Default Video":
    cap = cv2.VideoCapture('carPark.mp4')
else:
    uploaded_file = st.sidebar.file_uploader("Upload a video...", type=["mp4", "avi"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture("temp_video.mp4")
    else:
        cap = None

# Statut backend dans la sidebar
status_placeholder = st.sidebar.empty()
last_sync_placeholder = st.sidebar.empty()

if cap:
    st_frame = st.empty()
    last_send_time = time.time()

    while True:
        start_time = time.time()

        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Prétraitement image
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(
            imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 16
        )
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        # Détection des places
        img, etats = checkParkingSpace(img, imgDilate)

        # Envoi au backend selon l'intervalle
        current_time = time.time()
        if backend_enabled and (current_time - last_send_time) >= intervalle:
            ok, msg = envoyer_au_backend(etats)
            if ok:
                status_placeholder.success(msg)
            else:
                status_placeholder.warning(msg)
            last_sync_placeholder.caption(
                f"Dernière sync: {datetime.now().strftime('%H:%M:%S')}"
            )
            last_send_time = current_time

        # Affichage
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st_frame.image(img_rgb, channels="RGB")

        time.sleep(1 / fps_limit)