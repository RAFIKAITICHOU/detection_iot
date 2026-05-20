#!/usr/bin/env python3
"""
ParkiHna - Détection par Analyse de Pixels avec Affichage Visuel
Crée les places dans l'Admin Parking, dessine les frames en Rouge/Vert, et synchronise.
Idéal pour la vidéo carPark.mp4 où YOLOv8 a du mal avec la vue de haut.
"""

import cv2
import pickle
import numpy as np
import requests
import time
import os

# ==================== CONFIGURATION ====================
API_URL_LOGIN = "http://localhost:8081/api/auth/login"
API_URL_BATCH_PLACE = "http://localhost:8081/api/places/batch"
SYNC_API_URL = "http://localhost:8081/api/synchronisation/iot/synchroniser"
PARKING_ID = 1
BLOC = "0" # "0" car le parking Guichet a 1 seul étage (Niveau 0)
IP_RASPBERRY = "172.20.10.4"
ID_CAMERA = "CAM-01"

# Identifiants Admin pour la création automatique
EMAIL_ADMIN = "superadmin@smartparking.com"
PASSWORD_ADMIN = "admin123"

VIDEO_SOURCE = 'carPark.mp4'
INTERVALLE_DETECTION = 3  # Secondes
POSITIONS_FILE = 'CarParkPos' 

PLACE_WIDTH = 108   
PLACE_HEIGHT = 48
SEUIL_LIBRE = 900 # Nombre de pixels blancs max pour considérer libre

# ==================== GESTION DES PLACES ====================

def charger_et_creer_places():
    """Charge les positions depuis CarParkPos et crée les places dans le backend"""
    if not os.path.exists(POSITIONS_FILE):
        print(f"❌ Erreur: Le fichier {POSITIONS_FILE} n'existe pas.")
        print("Veuillez exécuter 'python parkingspacepicker.py' d'abord.")
        return None

    # Connexion à l'API pour récupérer un token
    token = ""
    try:
        print("[+] Connexion au Backend pour obtenir les droits d'administration...")
        res_login = requests.post(API_URL_LOGIN, json={"email": EMAIL_ADMIN, "motDePasse": PASSWORD_ADMIN})
        if res_login.status_code == 200:
            token = res_login.json().get('token', '')
            print("[OK] Connexion réussie !")
        else:
            print("[!] Échec de la connexion (Vérifiez les identifiants). Création ignorée.")
    except Exception as e:
        print("[!] Serveur inaccessible pour le login.")
        
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with open(POSITIONS_FILE, 'rb') as f:
        posList = pickle.load(f)
    
    print(f"✅ {len(posList)} places chargées depuis {POSITIONS_FILE}")
    places_formatees = []
    
    print("\n--- CREATION/SYNCHRONISATION DES PLACES AVEC LE BACKEND ---")
    
    # Résolution de la vidéo pour convertir en pourcentages (Frontend)
    VIDEO_WIDTH = 1100.0
    VIDEO_HEIGHT = 720.0
    
    places_payload = []
    
    for i, pos in enumerate(posList):
        place_id_local = i + 1
        x, y = pos
        numero = f"P-{place_id_local:03d}"
        
        # Le frontend utilise des pourcentages (0 à 100) pour dessiner sur le plan !
        perc_x = (x / VIDEO_WIDTH) * 100.0
        perc_y = (y / VIDEO_HEIGHT) * 100.0
        perc_w = (PLACE_WIDTH / VIDEO_WIDTH) * 100.0
        perc_h = (PLACE_HEIGHT / VIDEO_HEIGHT) * 100.0
        
        places_payload.append({
            "numero": numero,
            "type": "STANDARD",
            "statut": "LIBRE",
            "posX": perc_x, "posY": perc_y,
            "posW": perc_w, "posH": perc_h,
            "posR": 0
        })
        
        place_info = {
            "local_id": place_id_local,
            "numero": numero,
            "x": x, "y": y,
            "db_id": place_id_local  # Default fallback
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
                print(f"[OK] {len(data)} Places créées/mises à jour sur le Frontend en un seul batch !")
            elif response.status_code == 403:
                print(f"[!] Erreur 403 (Token invalide ou permissions manquantes)")
            else:
                print(f"[!] Erreur {response.status_code} lors de la création en batch.")
    except Exception as e:
        print(f"[!] Erreur lors de la synchronisation batch : {e}")
        
    return places_formatees

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
    return cv2.dilate(imgMedian, kernel, iterations=1)

def detecter_place(img_proc, x, y):
    zone = img_proc[y:y + PLACE_HEIGHT, x:x + PLACE_WIDTH]
    pixel_count = cv2.countNonZero(zone)
    # Si beaucoup de pixels blancs (contours), la place est OCCUPEE (1)
    return 1 if pixel_count > SEUIL_LIBRE else 0

# ==================== SYNCHRONISATION ====================

def envoyer_synchronisation(changements):
    if not changements: return
    payload = {
        "parkingId": PARKING_ID,
        "ipRaspberry": IP_RASPBERRY,
        "idCamera": ID_CAMERA,
        "detections": changements
    }
    try:
        requests.post(SYNC_API_URL, json=payload, timeout=10)
        print(f"📡 Backend mis à jour: {len(changements)} places.")
    except:
        pass

# ==================== BOUCLE PRINCIPALE ====================

def main():
    places = charger_et_creer_places()
    if not places: return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    derniers_etats = {p['db_id']: 0 for p in places} 
    last_sync_time = time.time()
    
    print("\n--- DÉMARRAGE DE L'ANALYSE ---")
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        img_proc = preprocess_image(frame)
        current_time = time.time()
        do_sync = (current_time - last_sync_time >= INTERVALLE_DETECTION)
        changements = []
        places_libres = 0
        
        for place in places:
            x, y = place['x'], place['y']
            db_id = place['db_id']
            
            # Détection (1 = Occupée, 0 = Libre)
            etat = detecter_place(img_proc, x, y)
            
            if etat == 0:
                places_libres += 1
                color = (0, 255, 0) # VERT
            else:
                color = (0, 0, 255) # ROUGE
            
            # Dessiner la frame (le carré) exactement sur les bornes de la place
            cv2.rectangle(frame, (x, y), (x + PLACE_WIDTH, y + PLACE_HEIGHT), color, 2)
            
            # Afficher le numéro de la place
            cv2.putText(frame, place['numero'], (x + 2, y + PLACE_HEIGHT - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Mémoriser les changements pour l'API
            if do_sync and derniers_etats.get(db_id) != etat:
                changements.append({"placeId": db_id, "etat": etat, "confidence": 0.95})
                derniers_etats[db_id] = etat
                print(f"🔄 Place {place['numero']} -> {'OCCUPEE' if etat==1 else 'LIBRE'}")
                
        if do_sync:
            envoyer_synchronisation(changements)
            last_sync_time = current_time
            
        # Montrer l'image avec les frames colorées
        cv2.imshow("ParkiHna - Visuel", frame)
        
        # Pour voir ce que l'algorithme "voit" (pixels blancs), décommentez la ligne suivante :
        # cv2.imshow("ParkiHna - Analyse", img_proc)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
