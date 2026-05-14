#!/usr/bin/env python3
"""
Parking Space Picker - Calibration des places de parking
Utilise une image fixe pour définir les zones des places
"""

import cv2
import pickle

# Dimensions des places (à ajuster selon votre image)
# Valeurs par défaut basées sur votre code original
WIDTH = 108   # 158 - 50
HEIGHT = 48   # 240 - 192

# Fichier de sauvegarde des positions
POSITIONS_FILE = "CarParkPos"

# Charger les positions existantes
try:
    with open(POSITIONS_FILE, 'rb') as f:
        posList = pickle.load(f)
    print(f"✓ {len(posList)} positions chargées")
except:
    posList = []
    print("✓ Nouveau fichier de positions créé")

# Charger l'image de référence
img_ref = cv2.imread('carParkImg.png')
if img_ref is None:
    print("❌ Erreur: carParkImg.png non trouvé!")
    print("   Assurez-vous que l'image est dans le même dossier")
    exit()

def mouseClick(events, x, y, flags, params):
    """Gestion des clics souris"""
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        print(f"✓ Place ajoutée à ({x}, {y}) - Total: {len(posList)}")
        
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + WIDTH and y1 < y < y1 + HEIGHT:
                posList.pop(i)
                print(f"✗ Place supprimée - Total: {len(posList)}")
                break
                
    # Sauvegarder immédiatement
    with open(POSITIONS_FILE, 'wb') as f:
        pickle.dump(posList, f)

# Créer la fenêtre
cv2.namedWindow('Parking Space Picker')
cv2.setMouseCallback('Parking Space Picker', mouseClick)

print("=" * 50)
print("🗺 PARKING SPACE CALIBRATION")
print("=" * 50)
print("INSTRUCTIONS:")
print("  - Cliquez GAUCHE: Ajouter une place")
print("  - Cliquez DROIT: Supprimer une place")
print("  - Appuyez sur 'q': Quitter et sauvegarder")
print("-" * 50)
print(f"Dimensions des places: {WIDTH} x {HEIGHT}")
print(f"Positions actuelles: {len(posList)}")
print("-" * 50)

while True:
    img_display = img_ref.copy()
    
    # Dessiner les places existantes
    for i, pos in enumerate(posList):
        x, y = pos
        cv2.rectangle(img_display, (x, y), (x + WIDTH, y + HEIGHT), (255, 0, 255), 2)
        cv2.putText(img_display, str(i+1), (x + 5, y + HEIGHT - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Afficher le nombre de places
    cv2.putText(img_display, f"Places: {len(posList)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Parking Space Picker', img_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("\n✅ Calibration terminée!")
print(f"   {len(posList)} places enregistrées dans {POSITIONS_FILE}")
