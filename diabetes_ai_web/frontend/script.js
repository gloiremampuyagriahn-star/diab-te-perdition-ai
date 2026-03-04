const API_BASE = "http://127.0.0.1:5000";

function setMessage(elementId, text, isError = false) {
  const element = document.getElementById(elementId);
  if (!element) return;
  element.textContent = text;
  element.style.color = isError ? "#dc2626" : "#059669";
}

async function registerUser() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();

  if (!username || !password) {
    setMessage("auth-message", "Nom d'utilisateur et mot de passe requis.", true);
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Échec de l'inscription");
    }

    setMessage("auth-message", data.message);
  } catch (error) {
    setMessage("auth-message", error.message, true);
  }
}

async function loginUser() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();

  if (!username || !password) {
    setMessage("auth-message", "Nom d'utilisateur et mot de passe requis.", true);
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || "Échec de la connexion");
    }

    localStorage.setItem("username", username);
    window.location.href = "dashboard.html";
  } catch (error) {
    setMessage("auth-message", error.message, true);
  }
}

function logoutUser() {
  localStorage.removeItem("username");
  localStorage.removeItem("lastPrediction");
  window.location.href = "login.html";
}

function getDashboardUser() {
  const username = localStorage.getItem("username");
  const welcomeText = document.getElementById("welcome-text");

  if (welcomeText) {
    if (!username) {
      window.location.href = "login.html";
      return null;
    }
    welcomeText.textContent = `Bienvenue, ${username}`;
  }

  return username;
}

function clearForm() {
  for (let i = 1; i <= 8; i++) {
    const field = document.getElementById(`f${i}`);
    if (field) field.value = "";
  }
  const message = document.getElementById("predict-message");
  if (message) message.textContent = "";
}

async function predictDiabetes() {
  const username = getDashboardUser();
  if (!username) return;

  const values = [];
  const labels = [
    "Grossesses",
    "Glucose",
    "Pression artérielle",
    "Épaisseur cutanée",
    "Insuline",
    "IMC",
    "Fonction diabète héréditaire",
    "Âge",
  ];

  for (let index = 1; index <= 8; index += 1) {
    const rawValue = document.getElementById(`f${index}`).value;
    const value = Number(rawValue);
    if (Number.isNaN(value) || rawValue === "") {
      setMessage("predict-message", "Veuillez remplir tous les champs.", true);
      return;
    }
    values.push(value);
  }

  const payload = {
    username,
    Pregnancies: values[0],
    Glucose: values[1],
    BloodPressure: values[2],
    SkinThickness: values[3],
    Insulin: values[4],
    BMI: values[5],
    DiabetesPedigreeFunction: values[6],
    Age: values[7],
  };

  try {
    setMessage("predict-message", "<i class='fas fa-spinner fa-spin'></i> Analyse en cours...");
    
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || data.error || "Échec de la prédiction");
    }

    // Sauvegarder les résultats pour la page result
    const resultData = {
      prediction: data.prediction,
      message: data.message,
      parameters: {
        "Nombre de grossesses": values[0],
        "Glucose (mg/dL)": values[1],
        "Pression artérielle (mmHg)": values[2],
        "Épaisseur cutanée (mm)": values[3],
        "Insuline (μU/mL)": values[4],
        "IMC (kg/m²)": values[5],
        "Fonction diabète héréditaire": values[6],
        "Âge (années)": values[7],
      },
      timestamp: new Date().toISOString(),
    };

    localStorage.setItem("lastPrediction", JSON.stringify(resultData));
    window.location.href = "result.html";
  } catch (error) {
    setMessage("predict-message", error.message, true);
  }
}

function displayResult() {
  const username = getDashboardUser();
  if (!username) return;

  const resultDataStr = localStorage.getItem("lastPrediction");
  if (!resultDataStr) {
    window.location.href = "dashboard.html";
    return;
  }

  const resultData = JSON.parse(resultDataStr);
  const resultCard = document.getElementById("result-card");
  const parametersGrid = document.getElementById("parameters-grid");

  const isPositive = resultData.prediction === 1;

  resultCard.className = `result-card ${isPositive ? "positive" : "negative"}`;
  resultCard.innerHTML = `
    <div class="result-icon">
      <i class="fas ${isPositive ? 'fa-exclamation-triangle' : 'fa-check-circle'}" style="font-size: 4rem;"></i>
    </div>
    <h2 class="result-title">${isPositive ? "Risque Détecté" : "Aucun Risque"}</h2>
    <p class="result-subtitle">${resultData.message || (isPositive ? "Le modèle indique un risque de diabète. Consultez un médecin pour confirmation." : "Le modèle n'indique pas de risque significatif de diabète actuellement.")}</p>
  `;

  let paramsHTML = "";
  for (const [label, value] of Object.entries(resultData.parameters)) {
    paramsHTML += `
      <div class="param-item">
        <div class="param-label">${label}</div>
        <div class="param-value">${value}</div>
      </div>
    `;
  }
  parametersGrid.innerHTML = paramsHTML;
}

async function loadStatistics() {
  const username = getDashboardUser();
  if (!username) return;

  try {
    const response = await fetch(`${API_BASE}/history/${username}`);
    const history = await response.json();

    if (!response.ok) {
      throw new Error("Erreur de chargement");
    }

    // Mettre à jour les statistiques
    const totalPredictions = history.length;
    const positiveCount = history.filter((h) => h.prediction === 1).length;
    const negativeCount = totalPredictions - positiveCount;

    document.getElementById("total-predictions").textContent = totalPredictions;
    document.getElementById("positive-count").textContent = positiveCount;
    document.getElementById("negative-count").textContent = negativeCount;

    // Créer le tableau d'historique
    const container = document.getElementById("history-table-container");

    if (history.length === 0) {
      container.innerHTML = '<p class="loading">Aucune analyse effectuée</p>';
      return;
    }

    let tableHTML = `
      <table class="history-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Glucose</th>
            <th>IMC</th>
            <th>Âge</th>
            <th>Résultat</th>
          </tr>
        </thead>
        <tbody>
    `;

    history.reverse().forEach((record) => {
      const date = record.created_at || new Date().toISOString();
      const formattedDate = new Date(date).toLocaleDateString("fr-FR");
      const isPositive = record.prediction === 1;

      tableHTML += `
        <tr>
          <td>${formattedDate}</td>
          <td>${record.glucose} mg/dL</td>
          <td>${record.bmi} kg/m²</td>
          <td>${record.age} ans</td>
          <td>
            <span class="badge ${isPositive ? "positive" : "negative"}">
              <i class="fas ${isPositive ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
              ${isPositive ? "Positif" : "Négatif"}
            </span>
          </td>
        </tr>
      `;
    });

    tableHTML += `
        </tbody>
      </table>
    `;

    container.innerHTML = tableHTML;
  } catch (error) {
    document.getElementById("history-table-container").innerHTML =
      '<p class="loading">Erreur de chargement des données</p>';
  }
}

getDashboardUser();
