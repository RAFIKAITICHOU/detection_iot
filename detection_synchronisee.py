#!/usr/bin/env python3
"""
Smart Parking - Detection IoT synchronisee avec reservations
Version amelioree qui communique avec l'API de synchronisation
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
# Configuration API
# IMPORTANT: Remplacez 'localhost' par l'adresse IP de votre PC si le script tourne sur le Raspberry Pi
API_URL = "http://localhost:8080/api/iot/detection"
SYNC_API_URL = "http://localhost:8080/api/synchronisation/iot/synchroniser"
PARKING_ID = 1
IP_RASPBERRY = "172.20.10.4"  # IP du Pi
ID_CAMERA = "CAM-01"

# Configuration video
VIDEO_SOURCE = 'carPark.mp4'
INTERVALLE_DETECTION = 3  # Secondes entre chaque analyse

# Dimensions des places (doivent correspondre a la calibration)
PLACE_WIDTH = 108   # 158 - 50
PLACE_HEIGHT = 48   # 240 - 192

# Seuil de detection
SEUIL_LIBRE = 900

# ==================== CHARGEMENT POSITIONS ====================
def load_positions():
    """Charger les positions des places depuis le fichier"""
    try:
        with open('CarParkPos', 'rb') as f:
            positions = pickle.load(f)
        print(f"OK: {len(positions)} places chargees")
        return positions
    except FileNotFoundError:
        print("Erreur: Fichier CarParkPos non trouve!")
        print("Lancez d'abord: python parkingspacepicker.py")
        return []
    except Exception as e:
        print(f"Erreur chargement: {e}")
        return []

# ==================== TRAITEMENT IMAGE ====================
def preprocess_image(img):
    """Pretraitement de l'image pour la detection"""
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
    """Detecte si une place est libre ou occupee"""
    zone = img_proc[y:y + PLACE_HEIGHT, x:x + PLACE_WIDTH]
    pixel_count = cv2.countNonZero(zone)
    
    if pixel_count < SEUIL_LIBRE:
        etat = 0  # Libre
        confidence = 1.0 - (pixel_count / SEUIL_LIBRE)
    else:
        etat = 1  # Occupee
        confidence = min(1.0, (pixel_count - SEUIL_LIBRE) / (PLACE_WIDTH * PLACE_HEIGHT - SEUIL_LIBRE))
    
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    return etat, confidence, pixel_count

# ==================== ENVOI API SYNCHRONISEE ====================
class DetectionSynchroniseeAPI:
    def __init__(self):
        self.derniers_etats = {}
        self.api_url = API_URL
        self.sync_api_url = SYNC_API_URL
        self.parking_id = PARKING_ID
        
    def envoyer_detections_synchronisees(self, detections):
        """Envoyer les detections a l'API de synchronisation Spring Boot"""
        if not detections:
            return True
            
        payload = {
            "parkingId": self.parking_id,
            "ipRaspberry": IP_RASPBERRY,
            "idCamera": ID_CAMERA,
            "detections": detections
        }
        
        try:
            # Utiliser l'API de synchronisation
            response = requests.post(
                self.sync_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Synchronisation reussie: {data.get('message', 'OK')}")
                return True
            else:
                print(f"Erreur API synchronisation: {response.status_code}")
                # Fallback sur l'API standard si synchronisation echoue
                return self.envoyer_detections_standard(detections)
                
        except requests.exceptions.ConnectionError:
            print(f"Impossible de se connecter a {self.sync_api_url}")
            # Fallback sur l'API standard
            return self.envoyer_detections_standard(detections)
        except Exception as e:
            print(f"Erreur: {e}")
            return False
    
    def envoyer_detections_standard(self, detections):
        """Fallback: envoyer les detections a l'API standard"""
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
                print(f"Envoi standard reussi")
                return True
            else:
                print(f"Erreur API standard: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Erreur envoi standard: {e}")
            return False
    
    def traiter_et_envoyer(self, positions, etats_detectes):
        """Comparer avec derniers etats et envoyer uniquement les changements"""
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
                print(f"Place {place_id}: {statut}")
        
        if changements:
            return self.envoyer_detections_synchronisees(changements)
        return True

# ==================== FONCTION PRINCIPALE ====================
def main():
    print("=" * 60)
    print("SMART PARKING - DETECTION SYNCHRONISEE")
    print("=" * 60)
    
    # 1. Charger les positions
    positions = load_positions()
    if not positions:
        return
    
    print(f"OK: Positions chargees: {len(positions)} places")
    
    # 2. Ouvrir la video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la video: {VIDEO_SOURCE}")
        return
    
    print(f"OK: Video chargee: {VIDEO_SOURCE}")
    
    # 3. Initialiser l'API
    api = DetectionSynchroniseeAPI()
    
    print("-" * 60)
    print(f"Parking ID: {PARKING_ID}")
    print(f"API URL: {API_URL}")
    print(f"Sync API URL: {SYNC_API_URL}")
    print(f"Intervalle: {INTERVALLE_DETECTION}s")
    print("-" * 60)
    print("Ctrl+C pour arreter")
    print("=" * 60)
    
    last_send_time = time.time()
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                # Boucler la video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Pretraitement de l'image
            img_proc = preprocess_image(img)
            
            # Detection pour chaque place
            etats = []
            
            for pos in positions:
                x, y = pos
                etat, confidence, pixel_count = detecter_place(img_proc, x, y)
                etats.append(etat)
            
            # Envoi a l'API a intervalle regulier
            current_time = time.time()
            if current_time - last_send_time >= INTERVALLE_DETECTION:
                api.traiter_et_envoyer(positions, etats)
                last_send_time = current_time
                
                # Afficher un resume periodique
                places_libres = etats.count(0)
                print(f"{datetime.now().strftime('%H:%M:%S')} - Libres: {places_libres}/{len(positions)}")
                
    except KeyboardInterrupt:
        print("\nArret demande...")
    finally:
        cap.release()
        print("Systeme arrete")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
