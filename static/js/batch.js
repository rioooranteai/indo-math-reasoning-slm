// Banyak soal page: import a CSV of questions, answer them one by one, export results.
(function () {
  const MAX_ROWS = 500;
  const QUESTION_COLS = ["soal", "question", "pertanyaan"];
  const KEY_COLS = ["kunci", "answer", "jawaban_benar", "kunci_jawaban"];

  const fileInput = document.getElementById("fileInput");
  const dropzone = document.getElementById("dropzone");
  const dropTitle = document.getElementById("dropTitle");
  const dropHint = document.getElementById("dropHint");
  const fileNotice = document.getElementById("fileNotice");
  const templateBtn = document.getElementById("templateBtn");
  const sampleBtn = document.getElementById("sampleBtn");
  const runBtn = document.getElementById("runBtn");
  const stopBtn = document.getElementById("stopBtn");
  const downloadBtn = document.getElementById("downloadBtn");
  const progress = document.getElementById("progress");
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const summary = document.getElementById("summary");
  const tableWrap = document.getElementById("tableWrap");
  const rowsEl = document.getElementById("rows");

  let rows = [];
  let delimiter = ",";
  let sourceName = "soal";
  let controller = null;

  // ---------- CSV ----------

  function detectDelimiter(text) {
    const header = text.split(/\r?\n/, 1)[0];
    const semis = (header.match(/;/g) || []).length;
    const commas = (header.match(/,/g) || []).length;
    return semis > commas ? ";" : ",";
  }

  function parseCsv(text, sep) {
    const out = [];
    let row = [];
    let field = "";
    let quoted = false;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (ch === '"') quoted = false;
        else field += ch;
      } else if (ch === '"') {
        quoted = true;
      } else if (ch === sep) {
        row.push(field); field = "";
      } else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); out.push(row); row = []; field = "";
      } else {
        field += ch;
      }
    }
    if (field !== "" || row.length) { row.push(field); out.push(row); }
    return out.filter((r) => r.some((cell) => cell.trim() !== ""));
  }

  function csvCell(value) {
    const text = value == null ? "" : String(value);
    return /[",;\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  }

  function toCsv(table, sep) {
    return "﻿" + table.map((r) => r.map(csvCell).join(sep)).join("\r\n") + "\r\n";
  }

  function download(name, content) {
    const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ---------- Loading rows ----------

  function showNotice(message) {
    fileNotice.textContent = message;
    fileNotice.hidden = !message;
  }

  function loadRows(list, label) {
    rows = list.map((r) => ({ question: r.question, key: r.key, answer: "", steps: "", time: null, status: "wait" }));
    dropTitle.textContent = label;
    dropHint.textContent = `${rows.length} soal siap dihitung.`;
    runBtn.disabled = rows.length === 0;
    runBtn.textContent = `Hitung ${rows.length} soal`;
    downloadBtn.disabled = true;
    progress.hidden = true;
    summary.hidden = true;
    renderTable();
  }

  function readCsvText(text, name) {
    delimiter = detectDelimiter(text);
    const table = parseCsv(text.replace(/^﻿/, ""), delimiter);
    if (table.length < 2) {
      showNotice("File ini kosong atau hanya berisi judul kolom. Isi minimal satu soal di bawah kolom soal.");
      return;
    }

    const header = table[0].map((h) => h.trim().toLowerCase());
    const qIndex = header.findIndex((h) => QUESTION_COLS.includes(h));
    const kIndex = header.findIndex((h) => KEY_COLS.includes(h));
    if (qIndex === -1) {
      showNotice("Kolom soal tidak ditemukan. Beri judul kolom pertama \"soal\", atau unduh templat CSV lalu salin soal ke dalamnya.");
      return;
    }

    let list = table.slice(1)
      .map((r) => ({ question: (r[qIndex] || "").trim(), key: kIndex === -1 ? "" : (r[kIndex] || "").trim() }))
      .filter((r) => r.question);

    if (!list.length) {
      showNotice("Kolom soal masih kosong. Isi minimal satu soal.");
      return;
    }

    let message = "";
    if (list.length > MAX_ROWS) {
      message = `File berisi ${list.length} soal. Hanya ${MAX_ROWS} soal pertama yang dipakai.`;
      list = list.slice(0, MAX_ROWS);
    }
    showNotice("");
    if (message) {
      fileNotice.textContent = message;
      fileNotice.classList.add("notice--info");
      fileNotice.hidden = false;
    } else {
      fileNotice.classList.remove("notice--info");
    }

    sourceName = name.replace(/\.csv$/i, "") || "soal";
    loadRows(list, name);
  }

  function handleFile(file) {
    if (!file) return;
    if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") {
      showNotice("File harus berformat CSV. Di Excel, pilih Simpan sebagai lalu CSV UTF-8.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => readCsvText(String(reader.result), file.name);
    reader.onerror = () => showNotice("File tidak dapat dibaca. Coba simpan ulang lalu unggah lagi.");
    reader.readAsText(file, "utf-8");
  }

  fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-over");
  }));
  ["dragleave", "drop"].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.remove("is-over")));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files[0]);
  });

  templateBtn.addEventListener("click", () => {
    download("templat-soal.csv", toCsv([
      ["soal", "kunci"],
      ["Ani punya 12 kelereng, lalu memberikan 5 kepada Budi. Berapa kelereng Ani sekarang?", "7"],
      ["Harga satu buku Rp4.000. Berapa harga 6 buku?", "24000"],
    ], ","));
  });

  sampleBtn.addEventListener("click", () => {
    const list = ["L1", "L2", "L3"].flatMap((level) =>
      (window.EXAMPLES[level] || []).map((e) => ({ question: e.q, key: e.a })));
    showNotice("");
    delimiter = ",";
    sourceName = "soal-contoh";
    loadRows(list, "Soal contoh GSM8K");
  });

  // ---------- Table ----------

  function statusCell(row) {
    if (row.status === "wait") return `<span class="status-wait">Menunggu</span>`;
    if (row.status === "running") return `<span class="status-wait">Menghitung…</span>`;
    if (row.status === "error") return `<span class="mark-bad">Gagal</span>`;
    if (!row.key) return "Selesai";
    return Format.sameAnswer(row.answer, row.key)
      ? `<span class="mark-ok">Benar</span>`
      : `<span class="mark-bad">Salah</span>`;
  }

  function escape(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function rowHtml(row, i) {
    return `<tr class="${row.status === "running" ? "is-running" : ""}">
      <td class="col-no">${i + 1}</td>
      <td><span class="q" title="${escape(row.question)}">${escape(row.question)}</span></td>
      <td class="col-num">${escape(row.key || "–")}</td>
      <td class="col-num">${escape(row.answer || "–")}</td>
      <td class="col-status">${statusCell(row)}</td>
    </tr>`;
  }

  function renderTable() {
    tableWrap.hidden = rows.length === 0;
    rowsEl.innerHTML = rows.map(rowHtml).join("");
  }

  function renderRow(i) {
    const tr = rowsEl.children[i];
    if (tr) tr.outerHTML = rowHtml(rows[i], i);
  }

  // ---------- Running ----------

  function setProgress(done) {
    const pct = rows.length ? (done / rows.length) * 100 : 0;
    progressFill.style.width = `${pct}%`;
    progressText.textContent = `${done} dari ${rows.length} soal selesai.`;
  }

  function showSummary(stopped) {
    const done = rows.filter((r) => r.status === "done");
    const graded = done.filter((r) => r.key);
    let text = stopped ? `Dihentikan. ${done.length} dari ${rows.length} soal sudah dijawab.` : `${done.length} soal sudah dijawab.`;
    if (graded.length) {
      const right = graded.filter((r) => Format.sameAnswer(r.answer, r.key)).length;
      const pct = Math.round((right / graded.length) * 100);
      text = `${right} dari ${graded.length} jawaban benar (${pct}%).` + (stopped ? " Perhitungan dihentikan sebelum selesai." : "");
    }
    summary.textContent = text;
    summary.hidden = false;
  }

  function setRunning(running) {
    runBtn.hidden = running;
    stopBtn.hidden = !running;
    sampleBtn.disabled = running;
    fileInput.disabled = running;
    downloadBtn.disabled = running || !rows.some((r) => r.status === "done" || r.status === "error");
  }

  runBtn.addEventListener("click", async () => {
    if (!rows.length) return;
    rows.forEach((r) => Object.assign(r, { answer: "", steps: "", time: null, status: "wait" }));
    renderTable();

    controller = new AbortController();
    progress.hidden = false;
    summary.hidden = true;
    showNotice("");
    setRunning(true);
    setProgress(0);

    let stopped = false;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      row.status = "running";
      renderRow(i);
      try {
        const data = await Api.infer(row.question, controller.signal);
        Object.assign(row, { answer: (data.answer || "").trim(), steps: data.chain_of_thought, time: data.elapsed_time, status: "done" });
      } catch (err) {
        if (err.name === "AbortError") {
          row.status = "wait";
          renderRow(i);
          stopped = true;
          break;
        }
        Object.assign(row, { answer: "", steps: err.message, status: "error" });
      }
      renderRow(i);
      setProgress(i + 1);
    }

    controller = null;
    setRunning(false);
    showSummary(stopped);
  });

  stopBtn.addEventListener("click", () => {
    if (controller) controller.abort();
  });

  downloadBtn.addEventListener("click", () => {
    const table = [["no", "soal", "kunci", "jawaban_model", "hasil", "waktu_detik", "langkah_penyelesaian"]];
    rows.forEach((r, i) => {
      let result = "";
      if (r.status === "error") result = "gagal";
      else if (r.status === "done") result = r.key ? (Format.sameAnswer(r.answer, r.key) ? "benar" : "salah") : "selesai";
      table.push([i + 1, r.question, r.key, r.answer, result, r.time == null ? "" : r.time, r.steps]);
    });
    download(`${sourceName}-hasil.csv`, toCsv(table, delimiter));
  });
})();
