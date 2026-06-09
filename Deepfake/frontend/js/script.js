const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const resultDiv = document.getElementById("result");
const placeholder = document.getElementById("placeholderText");
const clickableArea = document.getElementById("clickableArea");
const detectBtn = document.getElementById("detectBtn");

let selectedFile = null;
let previewUrl = null;
let inFlight = false;

detectBtn.disabled = false;

function showMessage(html) {
  resultDiv.innerHTML = html;
}

function setSelectedFile(file) {
  if (!file) return;

  if (!file.type || !file.type.startsWith("image/")) {
    showMessage("❗ File này không phải ảnh nha bà.");
    return;
  }

  selectedFile = file;
  detectBtn.disabled = false;

  // cleanup old preview url (tránh leak)
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.style.display = "block";
  placeholder.style.display = "none";

  showMessage(""); // clear kết quả cũ
}

async function runPredict() {
  if (!selectedFile) {
    showMessage("Vui lòng chọn ảnh trước khi kiểm tra.");
    return;
  }

  if (inFlight) return; // chặn spam khi đang chạy
  inFlight = true;

  detectBtn.disabled = true;
  showMessage("🔍 Đang phân tích...");

  try {
    const fd = new FormData();
    fd.append("image", selectedFile, selectedFile.name || "pasted.png");

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: fd,
      cache: "no-store"
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok || !data.ok) {
      showMessage("❗ Lỗi: " + (data.error || `HTTP ${res.status}`));
      return;
    }

    const realScore = data.real_score == null ? "N/A" : `${data.real_score}%`;
    const fakeScore = data.fake_score == null ? "N/A" : `${data.fake_score}%`;
    const ms = data.inference_ms == null ? "N/A" : `${data.inference_ms} ms`;

    showMessage(`
      <div class="result-title">Kết quả: <b>${data.label}</b></div>
      <div class="result-line">Real: <span class="score">${realScore}</span></div>
      <div class="result-line">Fake: <span class="score">${fakeScore}</span></div>
      <div class="result-line">Confidence: <span class="score">${data.confidence}%</span></div>
      <div class="result-line">Inference: <span class="score">${ms}</span></div>
    `);
  } catch (err) {
    showMessage("❗ Lỗi kết nối backend: " + (err.message || String(err)));
  } finally {
    detectBtn.disabled = false;
    inFlight = false;
  }
}

// click để chọn ảnh
clickableArea.addEventListener("click", () => imageInput.click());

// chọn file từ input
imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (file) setSelectedFile(file);
});

// bấm detect
detectBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  await runPredict();
});

// paste ảnh
document.addEventListener("paste", (event) => {
  const items = event.clipboardData && event.clipboardData.items;
  if (!items) return;

  for (const item of items) {
    if (item.type && item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) setSelectedFile(file);
      break;
    }
  }
});

// drag UI
["dragenter", "dragover"].forEach((evt) => {
  clickableArea.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    clickableArea.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((evt) => {
  clickableArea.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    clickableArea.classList.remove("drag-over");
  });
});

clickableArea.addEventListener("drop", (e) => {
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});
