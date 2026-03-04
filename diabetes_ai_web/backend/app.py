from pathlib import Path
import hashlib
import sys

import joblib
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add current directory to path for imports
try:
    BASE_DIR = Path(__file__).resolve().parent
except (NameError, RuntimeError):
    import database as db_module
    BASE_DIR = Path(db_module.__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database import SessionLocal, User, PredictionHistory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


MODELS_DIR = BASE_DIR / "models"


model = joblib.load(str(MODELS_DIR / "model.pkl"))
scaler = joblib.load(str(MODELS_DIR / "scaler.pkl"))


from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Sert les fichiers statiques du frontend
app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "frontend"), name="static")

# Redirige la racine vers le dashboard
@app.get("/")
def root():
    return RedirectResponse(url="/static/dashboard.html")


@app.post("/register")
def register(
    username: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user:
        raise HTTPException(status_code=400, detail="Utilisateur existe déjà")

    hashed = hash_password(password)
    new_user = User(username=username, password=hashed)
    db.add(new_user)
    db.commit()

    return {"message": "Utilisateur créé avec succès"}


@app.post("/login")
def login(
    username: str = Body(...),
    password: str = Body(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Utilisateur introuvable")

    if not verify_password(password, user.password):
        raise HTTPException(status_code=400, detail="Mot de passe incorrect")

    return {"message": "Connexion réussie", "username": username}


@app.post("/predict")
def predict(
    username: str = Body(...),
    Pregnancies: int = Body(...),
    Glucose: int = Body(...),
    BloodPressure: int = Body(...),
    SkinThickness: int = Body(...),
    Insulin: int = Body(...),
    BMI: float = Body(...),
    DiabetesPedigreeFunction: float = Body(...),
    Age: int = Body(...),
    db: Session = Depends(get_db),
):
    data = np.array(
        [
            [
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
            ]
        ]
    )


    data_scaled = scaler.transform(data)
    prediction = int(model.predict(data_scaled)[0])

    # Message personnalisé selon les valeurs et le résultat
    if prediction == 1:
        if Glucose > 125:
            message = "Risque élevé : taux de glucose élevé."
        elif BMI > 30:
            message = "Risque élevé : IMC élevé."
        elif Age > 50:
            message = "Risque élevé : âge avancé."
        else:
            message = "Risque détecté. Consultez un médecin."
    else:
        if Glucose < 90 and BMI < 25:
            message = "Aucun risque : taux de glucose et IMC normaux."
        else:
            message = "Aucun risque significatif détecté."

    history = PredictionHistory(
        username=username,
        glucose=Glucose,
        bmi=BMI,
        age=Age,
        prediction=prediction,
    )
    db.add(history)
    db.commit()

    return {"prediction": prediction, "message": message}


@app.get("/history/{username}")
def get_history(username: str, db: Session = Depends(get_db)):
    records = db.query(PredictionHistory).filter(
        PredictionHistory.username == username
    ).all()

    return [
        {
            "id": r.id,
            "username": r.username,
            "glucose": r.glucose,
            "bmi": r.bmi,
            "age": r.age,
            "prediction": r.prediction,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]
