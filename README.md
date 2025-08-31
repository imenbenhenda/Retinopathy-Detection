==============================
Retinopathy Detection Project
==============================

Projet : Détection automatique de la rétinopathie à partir d’images de la rétine.

Description :
-------------
Ce projet utilise le Transfer Learning avec des modèles pré-entraînés (ResNet50) 
pour classifier des images de rétines en deux catégories :
- 0 : sain
- 1 : rétinopathie

Le modèle est fine-tuné sur le dataset médical APTOS 2019 Blindness Detection.
https://www.kaggle.com/competitions/aptos2019-blindness-detection

Structure du projet :
--------------------
Retinopathy-Detection/
│── data/                 # Contiendra le dataset
│── notebooks/            # Jupyter notebooks pour exploration et tests
│   └── exploration.ipynb
│── src/                  # Code source principal
│   ├── train.py          # Script d’entraînement
│   ├── evaluate.py       # Script d’évaluation
│── models/               # Modèles entraînés et historiques
│   └── resnet50_best_finetune.pth
│── requirements.txt      # Librairies Python nécessaires
│── README.txt            # Documentation

Installation :
--------------
1. Cloner le dépôt :
   git clone <lien_du_projet>
   cd Retinopathy-Detection

2. Installer les dépendances :
   pip install -r requirements.txt

Utilisation :
-------------
Entraînement :
   python src/train.py
   - Phase 1 : entraînement de la tête (couche finale)
   - Phase 2 : fine-tuning des dernières couches

Évaluation :
   python src/evaluate.py
   - Affiche la loss, l’accuracy, le classification report et la matrice de confusion

Résultats attendus :
-------------------
- Validation Loss : ~0.0248
- Accuracy : 99.32%
- F1-score : ~0.993
- Matrice de confusion proche de la perfection :  
   Actual \ Predicted  0 (sain)  1 (malade)
   0 (sain)             369        2
   1 (malade)           3         358

Dataset :
---------
APTOS 2019 Blindness Detection (Kaggle)
Lien : https://www.kaggle.com/competitions/aptos2019-blindness-detection

Licence :
--------
Projet libre pour l'apprentissage et la recherche.
