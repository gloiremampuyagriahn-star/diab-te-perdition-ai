import hashlib
from database import SessionLocal, User

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

db = SessionLocal()
try:
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        hashed = hash_password("admin123")
        admin_user = User(username="admin", password=hashed)
        db.add(admin_user)
        db.commit()
        print("✓ Utilisateur admin créé avec succès")
    else:
        print("✓ Utilisateur admin existe déjà")
finally:
    db.close()
