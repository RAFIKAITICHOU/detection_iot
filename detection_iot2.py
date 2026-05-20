#!/usr/bin/env python3
"""
ParkiHna - Détection IoT (Version finale - filtrage des places valides)
"""

import cv2
import pickle
import numpy as np
import requests
import json
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
API_URL = "http://localhost:8081/api/iot/detection"
PARKING_ID = 1
IP_RASPBERRY = "172.20.10.4"
ID_CAMERA = "CAM-01"

VIDEO_SOURCE = 'carPark.mp4'
INTERVALLE_DETECTION = 2
PLACE_WIDTH = 108
PLACE_HEIGHT = 48
SEUIL_LIBRE = 900
LOT_SIZE = 15
RETRY_COUNT = 3

# ==================== CHARGEMENT POSITIONS ====================
def load_positions():
    try:
        with open('CarParkPos', 'rb') as f:
            positions = pickle.load(f)
        print(f"✓ {len(positions)} places chargées (fichier local)")
        return positions
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return []

# ==================== DÉCOUVERTE DES PLACES VALIDES ====================
def discover_valid_places(max_place_id=69):
    """Découvre quels placeId existent dans le backend"""
    print("🔍 Découverte des places valides dans le backend...")
    valid_ids = []
    
    for place_id in range(1, max_place_id + 1):
        payload = {
            "parkingId": PARKING_ID,
            "ipRaspberry": IP_RASPBERRY,
            "idCamera": ID_CAMERA,
            "detections": [{"placeId": place_id, "etat": 0, "confidence": 0.95}]
        }
        
        try:
            response = requests.post(
                API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "SUCCESS":
                    valid_ids.append(place_id)
                    print(f"  ✓ Place {place_id} existe", end="\r" if place_id % 10 != 0 else "\n")
                else:
                    print(f"  ✗ Place {place_id} n'existe pas")
            else:
                print(f"  ✗ Place {place_id} : HTTP {response.status_code}")
                
            time.sleep(0.05)  # Petite pause
            
        except Exception as e:
            print(f"  ✗ Place {place_id} : Erreur")
    
    print(f"\n📊 Résultat: {len(valid_ids)} places valides sur {max_place_id}")
    if valid_ids:
        print(f"   IDs: {valid_ids[:10]}{'...' if len(valid_ids) > 10 else ''}")
        print(f"   Plages: {valid_ids[0]}-{valid_ids[-1]}")
    
    return valid_ids

# ==================== TRAITEMENT IMAGE ====================
def preprocess_image(img):
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
    zone = img_proc[y:y + PLACE_HEIGHT, x:x + PLACE_WIDTH]
    pixel_count = cv2.countNonZero(zone)
    
    if pixel_count < SEUIL_LIBRE:
        etat = 0
        confidence = 1.0 - (pixel_count / SEUIL_LIBRE)
    else:
        etat = 1
        confidence = min(1.0, (pixel_count - SEUIL_LIBRE) / (PLACE_WIDTH * PLACE_HEIGHT - SEUIL_LIBRE))
    
    confidence = round(min(1.0, max(0.0, confidence)), 2)
    return etat, confidence

# ==================== ENVOI API ====================
class DetectionAPI:
    def __init__(self, valid_place_ids):
        self.valid_place_ids = valid_place_ids
        self.derniers_etats = {pid: None for pid in valid_place_ids}
        self.api_url = API_URL
        self.parking_id = PARKING_ID
        self.premier_envoi = True
        
        # Mapping: position index -> place_id (circulaire si plus de positions que d'IDs)
        self.position_to_placeid = {}
        for idx, pid in enumerate(valid_place_ids):
            if idx < 69:  # On a 69 positions dans CarParkPos
                self.position_to_placeid[idx] = pid
        
        print(f"✓ Mapping: {len(self.position_to_placeid)} positions → places valides")
        
    def envoyer_detections_par_lots(self, detections):
        if not detections:
            return True
        
        lots = [detections[i:i + LOT_SIZE] for i in range(0, len(detections), LOT_SIZE)]
        tous_succes = True
        
        for idx, lot in enumerate(lots):
            if len(lots) > 1:
                print(f"    Lot {idx+1}/{len(lots)} ({len(lot)} places)...", end=" ", flush=True)
            
            payload = {
                "parkingId": self.parking_id,
                "ipRaspberry": IP_RASPBERRY,
                "idCamera": ID_CAMERA,
                "detections": lot
            }
            
            succes = False
            for attempt in range(RETRY_COUNT):
                try:
                    response = requests.post(
                        self.api_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        if len(lots) > 1:
                            print("✓")
                        succes = True
                        break
                    else:
                        if attempt == RETRY_COUNT - 1 and len(lots) > 1:
                            print(f"✗ ({response.status_code})")
                        time.sleep(1)
                        
                except Exception as e:
                    if attempt == RETRY_COUNT - 1 and len(lots) > 1:
                        print(f"✗ (erreur)")
                    time.sleep(1)
            
            if not succes:
                tous_succes = False
                
        return tous_succes
    
    def traiter_et_envoyer(self, all_etats, all_confidences):
        """Traite uniquement les places qui ont un mapping valide"""
        
        # Construire les détections pour les places valides uniquement
        detections = []
        for idx, place_id in self.position_to_placeid.items():
            if idx < len(all_etats):
                detections.append({
                    "placeId": place_id,
                    "etat": int(all_etats[idx]),
                    "confidence": round(all_confidences[idx], 2)
                })
        
        if not detections:
            return True
        
        # Premier envoi: toutes les places valides
        if self.premier_envoi:
            print(f"  📡 Premier envoi: {len(detections)} places valides (lots de {LOT_SIZE})")
            succes = self.envoyer_detections_par_lots(detections)
            
            if succes:
                # Mémoriser les états
                for d in detections:
                    self.derniers_etats[d["placeId"]] = d["etat"]
                self.premier_envoi = False
                print("  ✓ État initial synchronisé avec le serveur")
            else:
                print("  ⚠️ Synchronisation initiale échouée, réessai...")
            
            return succes
        
        # Envois suivants: uniquement les changements
        changements = []
        for det in detections:
            place_id = det["placeId"]
            nouvel_etat = det["etat"]
            ancien_etat = self.derniers_etats.get(place_id)
            
            if ancien_etat != nouvel_etat:
                changements.append(det)
                self.derniers_etats[place_id] = nouvel_etat
                statut = "LIBRE" if nouvel_etat == 0 else "OCCUPEE"
                print(f"  Place {place_id}: {statut}")
        
        if changements:
            print(f"  📡 Envoi de {len(changements)} changements")
            return self.envoyer_detections_par_lots(changements)
        
        return True

# ==================== FONCTION PRINCIPALE ====================
def main():
    print("=" * 60)
    print("🚗 ParkiHna IoT DETECTION (Version finale)")
    print("=" * 60)
    
    # 1. Découvrir les places valides
    valid_place_ids = discover_valid_places(max_place_id=69)
    
    if not valid_place_ids:
        print("❌ Aucune place valide trouvée dans le backend!")
        print("   Vérifiez que le serveur est démarré et que des places sont configurées.")
        return
    
    # 2. Charger les positions
    positions = load_positions()
    if not positions:
        return
    
    # 3. Ouvrir la vidéo
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {VIDEO_SOURCE}")
        return
    
    print(f"✓ Vidéo chargée: {VIDEO_SOURCE}")
    
    # 4. Initialiser l'API
    api = DetectionAPI(valid_place_ids)
    
    print("-" * 60)
    print(f"Parking ID   : {PARKING_ID}")
    print(f"API URL      : {API_URL}")
    print(f"IP Raspberry : {IP_RASPBERRY}")
    print(f"Caméra       : {ID_CAMERA}")
    print(f"Intervalle   : {INTERVALLE_DETECTION}s")
    print(f"Places valides: {len(valid_place_ids)}")
    print(f"Taille lot   : {LOT_SIZE} places")
    print("-" * 60)
    print("Appuyez sur Ctrl+C pour arrêter")
    print("=" * 60)
    
    api = DetectionAPI(valid_place_ids)
    dernier_envoi = time.time()
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Traitement image
            img_proc = preprocess_image(img)
            
            # Détection pour toutes les positions
            etats = []
            confidences = []
            
            for pos in positions:
                x, y = pos
                etat, conf = detecter_place(img_proc, x, y)
                etats.append(etat)
                confidences.append(conf)
            
            # Envoi périodique
            current_time = time.time()
            if current_time - dernier_envoi >= INTERVALLE_DETECTION:
                api.traiter_et_envoyer(etats, confidences)
                
                # Compter les libres parmi les places valides
                places_libres = sum(1 for idx, pid in api.position_to_placeid.items() 
                                  if idx < len(etats) and etats[idx] == 0)
                print(f"⏺ {datetime.now().strftime('%H:%M:%S')} - "
                      f"Libres: {places_libres}/{len(api.position_to_placeid)}")
                
                dernier_envoi = current_time
                
    except KeyboardInterrupt:
        print("\n⏹ Arrêt demandé...")
    finally:
        cap.release()
        print("✅ Système arrêté proprement")

if __name__ == "__main__":
    main()
