from pathlib import Path
import joblib
import pandas as pd

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "diabetes_dataset_generated.csv"


def prepare_data():
    """Charge et nettoie le dataset, puis sépare les données"""
    print(f"Chargement du dataset depuis : {DATA_PATH}")
    
    # Charger le dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Afficher les valeurs null avant nettoyage
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Valeurs null trouvées :\n{null_counts[null_counts > 0]}")
    
    # Supprimer les lignes avec valeurs null
    df_cleaned = df.dropna()
    removed_rows = df.shape[0] - df_cleaned.shape[0]
    if removed_rows > 0:
        print(f"✓ {removed_rows} lignes avec valeurs null supprimées")
    print(f"Dataset nettoyé : {df_cleaned.shape[0]} lignes")
    
    # Séparer features (X) et target (y)
    # On suppose que la dernière colonne est le target (Outcome)
    X = df_cleaned.iloc[:, :-1].values
    y = df_cleaned.iloc[:, -1].values
    
    print(f"Features (X) : {X.shape}")
    print(f"Target (y) : {y.shape}")
    print(f"Distribution des classes : {np.bincount(y.astype(int))}")
    
    # Séparer en train/test
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_save():
    """Entraîne le modèle et sauvegarde les artifacts"""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DE DIABÈTE")
    print("="*60 + "\n")
    
    # Préparer les données
    x_train, x_test, y_train, y_test = prepare_data()
    
    print(f"\n📊 Données d'entraînement : {x_train.shape[0]} échantillons")
    print(f"📊 Données de test : {x_test.shape[0]} échantillons")
    
    # Normaliser les features
    print("\n⚙️ Normalisation des données...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Entraîner le modèle
    print("🤖 Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, verbose=1)
    model.fit(x_train_scaled, y_train)
    
    # Évaluer le modèle
    print("\n📈 Évaluation du modèle...")
    train_accuracy = model.score(x_train_scaled, y_train)
    test_accuracy = model.score(x_test_scaled, y_test)
    
    print(f"✓ Précision sur train : {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"✓ Précision sur test  : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Sauvegarder les artifacts
    print("\n💾 Sauvegarde du modèle et du scaler...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"✓ Modèle sauvegardé dans : {MODEL_PATH}")
    print(f"✓ Scaler sauvegardé dans : {SCALER_PATH}")
    
    print("\n" + "="*60)
    print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("="*60 + "\n")


if __name__ == "__main__":
    train_and_save()
