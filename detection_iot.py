#!/usr/bin/env python3
"""
Smart Parking - Détection IoT pour Raspberry Pi (Sans interface graphique)
Version headless - Pas d'affichage, seulement traitement et envoi API
"""

import cv2
import pickle
import numpy as np
import requests
import json
import time
import os
import sys
from datetime import datetime

# ==================== CONFIGURATION ====================
# Configuration API - À MODIFIER
API_URL = "http://172.20.10.3:8080/api/iot/detection"  # ← Changez l'IP
PARKING_ID = 1
IP_RASPBERRY = "172.20.10.4"  # ← IP de votre Pi
ID_CAMERA = "CAM-01"

# Configuration vidéo
VIDEO_SOURCE = 'carPark.mp4'
INTERVALLE_DETECTION = 2  # Secondes entre chaque analyse

# Dimensions des places (doivent correspondre à la calibration)
PLACE_WIDTH = 108   # 158 - 50
PLACE_HEIGHT = 48   # 240 - 192

# Seuil de détection
SEUIL_LIBRE = 900

# ==================== CHARGEMENT POSITIONS ====================
def load_positions():
    """Charger les positions des places depuis le fichier"""
    try:
        with open('CarParkPos', 'rb') as f:
            positions = pickle.load(f)
        print(f"✓ {len(positions)} places chargées")
        return positions
    except FileNotFoundError:
        print("❌ Fichier CarParkPos non trouvé!")
        print("   Lancez d'abord: python3 parkingspacepicker.py")
        return []
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return []

# ==================== TRAITEMENT IMAGE ====================
def preprocess_image(img):
    """Prétraitement de l'image pour la détection"""
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    return imgDilate

def detecter_place(img_proc, x, y):
    """Détecte si une place est libre ou occupée"""
    zone = img_proc[y:y + PLACE_HEIGHT, x:x + PLACE_WIDTH]
    pixel_count = cv2.countNonZero(zone)
    
    if pixel_count < SEUIL_LIBRE:
        etat = 0  # Libre
        confidence = 1.0 - (pixel_count / SEUIL_LIBRE)
    else:
        etat = 1  # Occupée
        confidence = min(1.0, (pixel_count - SEUIL_LIBRE) / (PLACE_WIDTH * PLACE_HEIGHT - SEUIL_LIBRE))
    
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    return etat, confidence, pixel_count

# ==================== ENVOI API ====================
class DetectionAPI:
    def __init__(self):
        self.derniers_etats = {}
        self.api_url = API_URL
        self.parking_id = PARKING_ID
        
    def envoyer_detections(self, detections):
        """Envoyer les détections à l'API Spring Boot"""
        if not detections:
            return True
            
        payload = {
            "parkingId": self.parking_id,
            "ipRaspberry": IP_RASPBERRY,
            "idCamera": ID_CAMERA,
            "detections": detections
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return True
            else:
                print(f"  ✗ Erreur API: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Impossible de se connecter à {self.api_url}")
            return False
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            return False
    
    def traiter_et_envoyer(self, positions, etats_detectes):
        """Comparer avec derniers états et envoyer uniquement les changements"""
        changements = []
        
        for i, pos in enumerate(positions):
            place_id = i + 1
            nouvel_etat = etats_detectes[i]
            ancien_etat = self.derniers_etats.get(place_id)
            
            if ancien_etat is None or ancien_etat != nouvel_etat:
                changements.append({
                    "placeId": place_id,
                    "etat": nouvel_etat,
                    "confidence": 0.95
                })
                self.derniers_etats[place_id] = nouvel_etat
                # Afficher le changement
                statut = "LIBRE" if nouvel_etat == 0 else "OCCUPEE"
                print(f"  Place {place_id}: {statut}")
        
        if changements:
            return self.envoyer_detections(changements)
        return True

# ==================== FONCTION PRINCIPALE ====================
def main():
    print("=" * 60)
    print("🚗 SMART PARKING IoT DETECTION (Headless Mode)")
    print("=" * 60)
    
    # 1. Charger les positions
    positions = load_positions()
    if not positions:
        return
    
    print(f"✓ Positions chargées: {len(positions)} places")
    
    # 2. Ouvrir la vidéo
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {VIDEO_SOURCE}")
        print("   Vérifiez que le fichier carPark.mp4 existe")
        return
    
    print(f"✓ Vidéo chargée: {VIDEO_SOURCE}")
    
    # 3. Initialiser l'API
    api = DetectionAPI()
    
    print("-" * 60)
    print(f"Parking ID: {PARKING_ID}")
    print(f"API URL: {API_URL}")
    print(f"Intervalle: {INTERVALLE_DETECTION}s")
    print(f"Seuil détection: {SEUIL_LIBRE}")
    print(f"Mode: Headless (pas d'affichage graphique)")
    print("-" * 60)
    print("Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)
    
    frame_count = 0
    last_send_time = time.time()
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                # Boucler la vidéo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Prétraitement de l'image
            img_proc = preprocess_image(img)
            
            # Détection pour chaque place
            etats = []
            
            for pos in positions:
                x, y = pos
                etat, confidence, pixel_count = detecter_place(img_proc, x, y)
                etats.append(etat)
            
            # Envoi à l'API à intervalle régulier
            current_time = time.time()
            if current_time - last_send_time >= INTERVALLE_DETECTION:
                api.traiter_et_envoyer(positions, etats)
                last_send_time = current_time
                
                # Afficher un résumé périodique
                places_libres = etats.count(0)
                print(f"⏺ {datetime.now().strftime('%H:%M:%S')} - "
                      f"Libres: {places_libres}/{len(positions)}")
                
    except KeyboardInterrupt:
        print("\n⏹ Arrêt demandé...")
    finally:
        cap.release()
        print("✅ Système arrêté")

if __name__ == "__main__":
    main()
