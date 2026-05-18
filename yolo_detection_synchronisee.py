#!/usr/bin/env python3
"""
Smart Parking - Détection avec YOLOv8 et Synchronisation
Utilise les places définies dans CarParkPos et YOLOv8 pour détecter les véhicules.
"""

import cv2
import pickle
import numpy as np
import requests
import time
import os
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
API_URL_LOGIN = "http://127.0.0.1:8080/api/auth/login"
API_URL_BATCH_PLACE = "http://127.0.0.1:8080/api/places/batch"
API_URL_GET_PLACES = "http://127.0.0.1:8080/api/places/parking/1"
SYNC_API_URL = "http://127.0.0.1:8080/api/synchronisation/iot/synchroniser"
PARKING_ID = 2
BLOC = "0"  # "0" car le parking Guichet a 1 seul étage (Niveau 0)
IP_RASPBERRY = "172.20.10.4"
ID_CAMERA = "CAM-01"

EMAIL_ADMIN = "superadmin@smartparking.com"
PASSWORD_ADMIN = "admin123"

VIDEO_SOURCE = 'carPark.mp4'
INTERVALLE_DETECTION = 3  # Secondes entre chaque analyse
POSITIONS_FILE = 'CarParkPos' # Le fichier généré par parkingspacepicker.py

PLACE_WIDTH = 108   # Dimensions de parkingspacepicker.py
PLACE_HEIGHT = 48

# Initialisation du modèle YOLOv8
print("Chargement du modèle YOLOv8...")
model = YOLO('yolov8n.pt')

# ==================== FONCTIONS UTILITAIRES ====================

def calculate_overlap_ratio(place_box, car_box):
    """Calcule le pourcentage de la place de parking qui est recouvert par la voiture"""
    x1 = max(place_box[0], car_box[0])
    y1 = max(place_box[1], car_box[1])
    x2 = min(place_box[2], car_box[2])
    y2 = min(place_box[3], car_box[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Surface de la place de parking
    place_area = (place_box[2] - place_box[0]) * (place_box[3] - place_box[1])
    
    if place_area == 0:
        return 0

    return inter_area / float(place_area)

def center_in_box(center_x, center_y, box):
    """Vérifie si le centre de la voiture est dans la place de parking"""
    return box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3]

# ==================== GESTION DES PLACES ====================

def charger_et_creer_places():
    """Charge les positions depuis CarParkPos et crée les places dans le backend en lot"""
    if not os.path.exists(POSITIONS_FILE):
        print(f"[!] Erreur: Le fichier {POSITIONS_FILE} n'existe pas.")
        print("Veuillez d'abord exécuter 'python parkingspacepicker.py' pour dessiner les places.")
        return None

    # Authentification pour obtenir le token
    token = ""
    try:
        print("[+] Connexion au Backend pour obtenir les droits d'administration...")
        res_login = requests.post(API_URL_LOGIN, json={"email": EMAIL_ADMIN, "motDePasse": PASSWORD_ADMIN})
        if res_login.status_code == 200:
            token = res_login.json().get('token', '')
            print("[OK] Connexion réussie !")
        else:
            print("[!] Échec de la connexion (Vérifiez les identifiants).")
    except Exception as e:
        print("[!] Serveur inaccessible pour le login.")
        
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # 1. Récupérer les places existantes depuis le serveur pour éviter de recréer
    existing_places_by_num = {}
    try:
        if token:
            print("[+] Récupération des places existantes du serveur pour éviter les doublons...")
            res_get = requests.get(API_URL_GET_PLACES, headers=headers, timeout=10)
            if res_get.status_code == 200:
                list_existing = res_get.json()
                for p in list_existing:
                    # Associer par numéro de place (ex: P-001) dans le bon bloc/étage
                    if p.get('numero') and (p.get('bloc') == BLOC or p.get('numeroEtage') == int(BLOC)):
                        existing_places_by_num[p['numero']] = p
                print(f"[OK] {len(existing_places_by_num)} places existantes chargées pour le bloc {BLOC}.")
    except Exception as e:
        print(f"[!] Impossible de charger les places existantes (fallback création brute) : {e}")

    with open(POSITIONS_FILE, 'rb') as f:
        posList = pickle.load(f)
    
    print(f"[OK] {len(posList)} places chargées depuis {POSITIONS_FILE}")
    places_formatees = []
    places_payload = []
    
    VIDEO_WIDTH = 1100.0
    VIDEO_HEIGHT = 720.0
    
    print("\n--- SYNCHRONISATION DES PLACES AVEC LE BACKEND ---")
    for i, pos in enumerate(posList):
        place_id_local = i + 1
        x, y = pos
        w, h = PLACE_WIDTH, PLACE_HEIGHT
        x2, y2 = x + w, y + h
        
        numero = f"P-{place_id_local:03d}"
        
        # Le frontend utilise des pourcentages (0 à 100) pour dessiner sur le plan
        perc_x = (x / VIDEO_WIDTH) * 100.0
        perc_y = (y / VIDEO_HEIGHT) * 100.0
        perc_w = (w / VIDEO_WIDTH) * 100.0
        perc_h = (h / VIDEO_HEIGHT) * 100.0
        
        # Vérifier si la place existe déjà par son numéro
        exist = existing_places_by_num.get(numero)
        db_id = exist['id'] if exist else None
        statut_existant = exist['statut'] if exist else "LIBRE"
        
        place_payload_item = {
            "numero": numero,
            "type": "STANDARD",
            "statut": statut_existant, # Préserver le statut existant (ex: RESERVEE)
            "posX": perc_x, "posY": perc_y,
            "posW": perc_w, "posH": perc_h,
            "posR": 0
        }
        if db_id is not None:
            place_payload_item["id"] = db_id # Passer l'ID existant pour faire un UPDATE au lieu d'un CREATE
            
        places_payload.append(place_payload_item)
        
        place_info = {
            "local_id": place_id_local,
            "numero": numero,
            "x1": x, "y1": y, "x2": x2, "y2": y2,
            "db_id": db_id if db_id is not None else place_id_local
        }
        places_formatees.append(place_info)
        
    try:
        if token:
            payload = {
                "parkingId": PARKING_ID,
                "bloc": BLOC,
                "places": places_payload
            }
            response = requests.put(API_URL_BATCH_PLACE, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for i, p_res in enumerate(data):
                    if i < len(places_formatees):
                        places_formatees[i]['db_id'] = p_res.get('id', places_formatees[i]['local_id'])
                print(f"[OK] {len(data)} Places synchronisées avec succès (Création/Mise à jour préservant les états) !")
            elif response.status_code == 403:
                print(f"[!] Erreur 403 (Token invalide ou permissions manquantes)")
            else:
                print(f"[!] Erreur {response.status_code} lors de la création en batch.")
    except Exception as e:
        print(f"[!] Erreur lors de la synchronisation batch : {e}")
        
    return places_formatees

# ==================== SYNCHRONISATION ====================

def fetch_backend_states(token):
    """Récupère l'état réel des places depuis le backend (LIBRE, OCCUPEE, RESERVEE)"""
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(API_URL_GET_PLACES, headers=headers, timeout=5)
        if response.status_code == 200:
            places_data = response.json()
            return {p['id']: p['statut'] for p in places_data}
    except Exception as e:
        pass
    return None

def envoyer_synchronisation(changements):
    """Envoie les changements d'état à l'API de synchronisation"""
    if not changements:
        return
        
    payload = {
        "parkingId": PARKING_ID,
        "ipRaspberry": IP_RASPBERRY,
        "idCamera": ID_CAMERA,
        "detections": changements
    }
    
    try:
        response = requests.post(SYNC_API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            pass # Silent success for log clarity
        else:
            print(f"[!] Erreur synchronisation: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[!] Erreur de connexion API synchronisation: {e}")

# ==================== BOUCLE PRINCIPALE ====================

def main():
    places = charger_et_creer_places()
    if not places:
        return

    # S'authentifier une fois pour le fetch des statuts
    token = ""
    try:
        res = requests.post(API_URL_LOGIN, json={"email": EMAIL_ADMIN, "motDePasse": PASSWORD_ADMIN})
        if res.status_code == 200:
            token = res.json().get('token', '')
    except:
        pass

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {VIDEO_SOURCE}")
        return

    derniers_etats_iot = {p['db_id']: 0 for p in places} 
    etats_backend = {p['db_id']: "LIBRE" for p in places} # Stocke l'état réel (LIBRE, OCCUPEE, RESERVEE)
    last_sync_time = time.time() - INTERVALLE_DETECTION
    last_fetch_time = time.time() - INTERVALLE_DETECTION
    
    print("\n--- DÉMARRAGE DE LA DÉTECTION TEMPS RÉEL (YOLOv8) ---")
    
    voitures_boxes = [] # Pour stocker les détections et les dessiner à chaque frame
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        current_time = time.time()
        
        # Copie de l'image pour l'affichage
        img_display = frame.copy()
        
        # Analyser l'image toutes les X secondes
        if current_time - last_sync_time >= INTERVALLE_DETECTION:
            # 1. Détecter toutes les voitures avec YOLOv8
            results = model(frame, conf=0.10, verbose=False)
            voitures_boxes = []
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    # Classes COCO: 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
                    if cls_id in [2, 3, 5, 7]: 
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        voitures_boxes.append([x1, y1, x2, y2])
            
            # Prétraitement robuste pour la méthode de comptage de pixels (vue de dessus de carPark.mp4)
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(
                imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 25, 16
            )
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            
            changements = []
            places_libres = 0
            
            # 2. Vérifier chaque place avec le double détecteur (YOLO + Pixel Count)
            for place in places:
                place_box = [place['x1'], place['y1'], place['x2'], place['y2']]
                place_id_db = place['db_id']
                
                # A. Détection avec le comptage de pixels (100% robuste sur cette vidéo vue de dessus)
                zone = imgDilate[place['y1']:place['y2'], place['x1']:place['x2']]
                pixel_count = cv2.countNonZero(zone)
                est_occupee_pixels = pixel_count >= 900
                
                # B. Détection avec YOLOv8 (si YOLO détecte des voitures)
                est_occupee_yolo = False
                for car_box in voitures_boxes:
                    overlap = calculate_overlap_ratio(place_box, car_box)
                    car_center_x = (car_box[0] + car_box[2]) / 2
                    car_center_y = (car_box[1] + car_box[3]) / 2
                    
                    if overlap > 0.30 or center_in_box(car_center_x, car_center_y, place_box):
                        est_occupee_yolo = True
                        break
                
                # La place est occupée si l'une ou l'autre méthode le détecte
                est_occupee = est_occupee_pixels or est_occupee_yolo
                
                nouvel_etat = 1 if est_occupee else 0
                if nouvel_etat == 0:
                    places_libres += 1
                
                # Enregistrer le changement pour l'envoi IoT
                if derniers_etats_iot.get(place_id_db) != nouvel_etat:
                    changements.append({
                        "placeId": place_id_db,
                        "etat": nouvel_etat,
                        "confidence": 0.95
                    })
                    derniers_etats_iot[place_id_db] = nouvel_etat
                    
                    statut = "OCCUPEE" if nouvel_etat == 1 else "LIBRE"
                    print(f"🔄 Détection : Place {place['numero']} -> {statut} (Pixels: {pixel_count})")
            
            print(f"📊 Analyse terminée - Places libres: {places_libres}/{len(places)}")
            
            # Mettre à jour le backend avec les détections
            if changements:
                envoyer_synchronisation(changements)
                
            last_sync_time = current_time

        # Récupérer l'état réel du backend (pour voir les places RESERVEE) toutes les 3 secondes
        if current_time - last_fetch_time >= 3.0:
            backend_states = fetch_backend_states(token)
            if backend_states:
                for db_id, statut in backend_states.items():
                    etats_backend[db_id] = statut
            last_fetch_time = current_time

        # Affichage visuel (Mise à jour à chaque frame)
        # 1. Dessiner les voitures détectées par YOLOv8 en BLEU
        for car_box in voitures_boxes:
            cv2.rectangle(img_display, (car_box[0], car_box[1]), (car_box[2], car_box[3]), (255, 0, 0), 1)

        # 2. Dessiner les places de parking avec la couleur du Backend !
        for place in places:
            statut = etats_backend.get(place['db_id'], "LIBRE")
            
            # (B, G, R) dans OpenCV
            if statut == "OCCUPEE":
                color = (0, 0, 255)     # ROUGE
            elif statut == "RESERVEE":
                color = (0, 255, 255)   # JAUNE
            else:
                color = (0, 255, 0)     # VERT
            
            cv2.rectangle(img_display, (place['x1'], place['y1']), (place['x2'], place['y2']), color, 2)
            cv2.putText(img_display, str(place['numero']), (place['x1'], place['y1']-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.imshow("Smart Parking - YOLOv8", img_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[+] Détection arrêtée proprement par l'utilisateur.")
