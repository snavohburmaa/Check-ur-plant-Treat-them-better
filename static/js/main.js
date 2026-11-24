const PLANTS = [
    { id: "Apple", icon: "\u{1F34E}" },
    { id: "Blueberry", icon: "\u{1FAD0}" },
    { id: "Cherry", icon: "\u{1F352}" },
    { id: "Corn", icon: "\u{1F33D}" },
    { id: "Grape", icon: "\u{1F347}" },
    { id: "Orange", icon: "\u{1F34A}" },
    { id: "Peach", icon: "\u{1F351}" },
    { id: "Bell Pepper", icon: "\u{1FAD1}" },
    { id: "Potato", icon: "\u{1F954}" },
    { id: "Raspberry", icon: "\u{1FAD0}" },
    { id: "Soybean", icon: "\u{1F331}" },
    { id: "Squash", icon: "\u{1F383}" },
    { id: "Strawberry", icon: "\u{1F353}" },
    { id: "Tomato", icon: "\u{1F345}" },
];

let chosenPlant = null;

const stepPlant = document.getElementById("step-plant");
const stepUpload = document.getElementById("step-upload");
const stepResults = document.getElementById("step-results");
const plantGrid = document.getElementById("plant-grid");
const chosenPlantName = document.getElementById("chosen-plant-name");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewArea = document.getElementById("preview-area");
const diagCard = document.getElementById("diagnosis-card");
const altResults = document.getElementById("alt-results");
const scanOverlay = document.getElementById("scan-overlay");
const scanImg = document.getElementById("scan-img");
const backBtn = document.getElementById("back-btn");
const restartBtn = document.getElementById("restart-btn");

function showStep(step) {
    [stepPlant, stepUpload, stepResults].forEach((s) => s.classList.remove("active"));
    step.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
}

PLANTS.forEach((plant) => {
    const card = document.createElement("div");
    card.className = "plant-card";
    card.tabIndex = 0;
    card.innerHTML = `<span class="plant-icon">${plant.icon}</span><span class="plant-name">${plant.id}</span>`;
    card.addEventListener("click", () => selectPlant(plant));
    card.addEventListener("keydown", (e) => { if (e.key === "Enter") selectPlant(plant); });
    plantGrid.appendChild(card);
});

function selectPlant(plant) {
    chosenPlant = plant.id;
    chosenPlantName.textContent = plant.id;
    resetUpload();
    showStep(stepUpload);
}

backBtn.addEventListener("click", () => { chosenPlant = null; showStep(stepPlant); });
restartBtn.addEventListener("click", () => { chosenPlant = null; showStep(stepPlant); });

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length > 0) handleFile(fileInput.files[0]); });

function resetUpload() {
    previewArea.innerHTML = `
        <div class="upload-prompt">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            <p>Drop a leaf photo here</p>
            <span class="hint">or click to browse</span>
        </div>`;
    fileInput.value = "";
}

function handleFile(file) {
    if (!file.type.startsWith("image/")) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewArea.innerHTML = `<img src="${e.target.result}" alt="Leaf photo">`;
        scanImg.src = e.target.result;
        scanOverlay.classList.add("visible");
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);
    if (chosenPlant) formData.append("plant", chosenPlant);

    fetch("/diagnose", { method: "POST", body: formData })
        .then((r) => r.json())
        .then((data) => {
            setTimeout(() => {
                scanOverlay.classList.remove("visible");
                if (data.ok) {
                    showDiagnosis(data.results);
                    showStep(stepResults);
                } else {
                    const toast = document.createElement("div");
                    toast.className = "error-toast";
                    toast.textContent = data.error || "Diagnosis failed. Try again.";
                    previewArea.parentElement.after(toast);
                    setTimeout(() => toast.remove(), 5000);
                }
            }, 2000);
        })
        .catch(() => {
            scanOverlay.classList.remove("visible");
        });
}

function showDiagnosis(results) {
    const top = results[0];
    const isHealthy = top.disease.toLowerCase() === "healthy";

    const severityClass = {
        None: "severity-none",
        Moderate: "severity-moderate",
        High: "severity-high",
        Critical: "severity-critical",
    }[top.severity] || "severity-moderate";

    let treatmentHTML = "";
    if (top.treatment && top.treatment.length > 0) {
        treatmentHTML = `
            <div class="diag-section">
                <h4>${isHealthy ? "Care tips" : "Treatment"}</h4>
                <ul class="treatment-list">
                    ${top.treatment.map((t) => `<li>${t}</li>`).join("")}
                </ul>
            </div>`;
    }

    let preventionHTML = "";
    if (top.prevention && top.prevention.length > 0) {
        preventionHTML = `
            <div class="diag-section">
                <h4>Prevention</h4>
                <ul class="treatment-list">
                    ${top.prevention.map((t) => `<li>${t}</li>`).join("")}
                </ul>
            </div>`;
    }

    let summaryHTML = "";
    if (top.summary) {
        summaryHTML = `<div class="diag-section"><p class="ai-summary">${top.summary}</p></div>`;
    }

    diagCard.innerHTML = `
        <div class="diag-header">
            <div>
                <div class="diag-title">${top.disease}</div>
                <div class="diag-plant">${top.plant}</div>
            </div>
            <span class="severity-badge ${severityClass}">${top.severity === "None" ? "Healthy" : top.severity}</span>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-label">Confidence ${top.confidence}%</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width:0%"></div>
            </div>
        </div>
        ${summaryHTML}
        <div class="diag-section">
            <h4>Cause</h4>
            <p>${top.cause}</p>
        </div>
        ${top.symptoms ? `<div class="diag-section"><h4>Symptoms</h4><p>${top.symptoms}</p></div>` : ""}
        ${treatmentHTML}
        ${preventionHTML}`;

    requestAnimationFrame(() => {
        const bar = diagCard.querySelector(".confidence-bar");
        if (bar) bar.style.width = top.confidence + "%";
    });

    if (results.length > 1) {
        altResults.innerHTML =
            `<h3>Other possibilities</h3>` +
            results.slice(1).map((r) => `
                <div class="alt-card">
                    <span class="alt-name">${r.plant} — ${r.disease}</span>
                    <span class="alt-conf">${r.confidence}%</span>
                </div>`).join("");
    } else {
        altResults.innerHTML = "";
    }
}
