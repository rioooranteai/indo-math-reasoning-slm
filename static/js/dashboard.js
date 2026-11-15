// Endpoint relatif: karena frontend ini dimount dan disajikan oleh
// FastAPI yang sama, tidak perlu origin absolut/hardcoded lagi.
const INFERENCE_ENDPOINT = "/infer";

const EXAMPLES = {
  L1: "Berapa hasil dari 24 + 18?",
  L2: "Sebuah persegi panjang memiliki panjang 12 cm dan lebar 7 cm. Berapa luasnya?",
  L3: "Jika 3x^2 - 5x - 2 = 0, tentukan nilai x yang memenuhi persamaan tersebut."
};

const questionInput = document.getElementById('questionInput');
const runBtn = document.getElementById('runBtn');
const finalAnswer = document.getElementById('finalAnswer');
const cotBox = document.getElementById('cotBox');
const cotToggle = document.getElementById('cotToggle');
const metaModel = document.getElementById('metaModel');
const metaTime = document.getElementById('metaTime');
const metaTimestamp = document.getElementById('metaTimestamp');
const historyList = document.getElementById('historyList');
const statTotal = document.getElementById('statTotal');
const statAvg = document.getElementById('statAvg');
const themeToggle = document.getElementById('themeToggle');

let sessionStats = { total: 0, totalTime: 0 };
let history = [];

document.querySelectorAll('.tag-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    questionInput.value = EXAMPLES[btn.dataset.level];
  });
});

cotToggle.addEventListener('click', () => {
  const visible = cotBox.style.display !== 'none';
  cotBox.style.display = visible ? 'none' : 'block';
  cotToggle.textContent = visible ? '▸ Tampilkan chain-of-thought' : '▾ Sembunyikan chain-of-thought';
});

themeToggle.addEventListener('click', () => {
  const html = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  themeToggle.textContent = isDark ? 'Mode Gelap' : 'Mode Terang';
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
});

(function initTheme(){
  const saved = localStorage.getItem('theme');
  if (saved === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeToggle.textContent = 'Mode Terang';
  }
})();

async function callInferenceEndpoint(question){
  const response = await fetch(INFERENCE_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!response.ok) {
    let detail = 'Endpoint mengembalikan status ' + response.status;
    try {
      const errBody = await response.json();
      if (errBody.detail) detail = errBody.detail;
    } catch (_) { /* respons bukan JSON, pakai pesan default */ }
    throw new Error(detail);
  }
  return await response.json();
}

function renderStats(){
  statTotal.textContent = sessionStats.total;
  const avg = sessionStats.total ? (sessionStats.totalTime / sessionStats.total) : 0;
  statAvg.textContent = avg.toFixed(2) + 's';
}

function renderHistory(){
  if (history.length === 0){
    historyList.innerHTML = '<p class="empty-note">Belum ada riwayat inferensi.</p>';
    return;
  }
  historyList.innerHTML = history.map((h, i) => `
    <div class="history-item" data-idx="${i}">
      <span>${h.question.slice(0,60)}${h.question.length>60?'...':''}</span>
      <span>${h.timestamp}</span>
    </div>
  `).join('');
  historyList.querySelectorAll('.history-item').forEach(el => {
    el.addEventListener('click', () => {
      const item = history[el.dataset.idx];
      questionInput.value = item.question;
      showResult(item);
    });
  });
}

function showResult(result){
  finalAnswer.textContent = result.answer || '(tidak ditemukan)';
  cotBox.textContent = result.chain_of_thought || '(tidak ada chain-of-thought)';
  metaModel.textContent = result.model || '-';
  metaTime.textContent = result.elapsed_time.toFixed(2) + 's';
  metaTimestamp.textContent = result.timestamp;
}

runBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  if (!question){
    alert('Masukkan soal matematika terlebih dahulu.');
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = 'Sedang menjalankan inferensi...';
  finalAnswer.textContent = '...';
  cotBox.textContent = 'Menunggu respons dari model...';

  try {
    const data = await callInferenceEndpoint(question);

    const result = {
      question,
      model: data.model,
      answer: data.answer,
      chain_of_thought: data.chain_of_thought,
      elapsed_time: data.elapsed_time,
      timestamp: data.timestamp
    };

    sessionStats.total += 1;
    sessionStats.totalTime += result.elapsed_time;

    history.unshift(result);
    if (history.length > 10) history.pop();

    showResult(result);
    renderStats();
    renderHistory();
  } catch (err) {
    finalAnswer.textContent = 'Gagal terhubung ke model.';
    cotBox.textContent = 'Error: ' + err.message + '\n\nPastikan server FastAPI sudah selesai memuat model (cek /health) sebelum mencoba lagi.';
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = 'Jalankan Inferensi';
  }
});

renderStats();
renderHistory();
