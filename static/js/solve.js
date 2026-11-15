// Hitung page: solve one question and show the answer with its steps.
(function () {
  const form = document.getElementById("askForm");
  const question = document.getElementById("question");
  const solveBtn = document.getElementById("solveBtn");
  const formNotice = document.getElementById("formNotice");
  const result = document.getElementById("result");
  const waiting = document.getElementById("waiting");
  const waitingText = document.getElementById("waitingText");
  const answerBlock = document.getElementById("answerBlock");
  const emptyState = document.getElementById("emptyState");
  const answer = document.getElementById("answer");
  const answerMeta = document.getElementById("answerMeta");
  const steps = document.getElementById("steps");
  const stepsBody = document.getElementById("stepsBody");

  // Answer key of the example currently in the box, if any.
  let exampleKey = null;
  let busy = false;
  const lastPick = {};

  // Let the question box grow with its text, like writing on paper.
  function autosize() {
    question.style.height = "auto";
    question.style.height = `${question.scrollHeight}px`;
  }

  function showNotice(message) {
    formNotice.textContent = message;
    formNotice.hidden = !message;
  }

  function pickExample(level) {
    const list = window.EXAMPLES[level] || [];
    if (!list.length) return null;
    let index = Math.floor(Math.random() * list.length);
    if (list.length > 1 && index === lastPick[level]) index = (index + 1) % list.length;
    lastPick[level] = index;
    return list[index];
  }

  const cards = document.querySelectorAll(".example-card");

  function markPicked(card) {
    cards.forEach((c) => c.setAttribute("aria-pressed", String(c === card)));
  }

  cards.forEach((card) => {
    card.setAttribute("aria-pressed", "false");
    card.addEventListener("click", () => {
      const example = pickExample(card.dataset.level);
      if (!example) return;
      question.value = example.q;
      exampleKey = example.a;
      markPicked(card);
      autosize();
      showNotice("");
      question.focus();
    });
  });

  question.addEventListener("input", () => {
    exampleKey = null;
    markPicked(null);
    autosize();
  });

  window.addEventListener("resize", autosize);

  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  function setEmpty(empty) {
    result.classList.toggle("is-empty", empty);
    emptyState.hidden = !empty;
  }

  // Bring the answer row into view when it starts below the fold (small screens).
  function revealResult() {
    if (result.getBoundingClientRect().top < window.innerHeight * 0.6) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    result.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  }

  function startWaiting() {
    const started = performance.now();
    setEmpty(false);
    answerBlock.hidden = true;
    waiting.hidden = false;
    waitingText.textContent = "Sedang menghitung…";
    revealResult();

    const timer = setInterval(() => {
      const secs = Math.floor((performance.now() - started) / 1000);
      waitingText.textContent = `Sedang menghitung… ${secs} detik`;
    }, 1000);
    return () => clearInterval(timer);
  }

  function span(text, className) {
    const el = document.createElement("span");
    el.textContent = text;
    if (className) el.className = className;
    return el;
  }

  function showAnswer(data) {
    const text = (data.answer || "").trim() || "Tidak ditemukan";
    answer.textContent = text;
    answer.classList.toggle("answer--long", text.length > 8);
    // Restart the reveal animation for every new answer.
    answer.classList.remove("answer-reveal");
    void answer.offsetWidth;
    answer.classList.add("answer-reveal");

    const meta = [span(`Selesai dalam ${Format.seconds(data.elapsed_time)} detik.`)];
    if (exampleKey !== null) {
      meta.push(Format.sameAnswer(text, exampleKey)
        ? span("Sesuai kunci jawaban.", "mark-ok")
        : span(`Kunci jawabannya ${exampleKey}.`, "mark-bad"));
    }
    answerMeta.replaceChildren(...meta);

    stepsBody.innerHTML = Steps.render(data.chain_of_thought);
    steps.open = false;
    waiting.hidden = true;
    answerBlock.hidden = false;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;

    const text = question.value.trim();
    if (!text) {
      showNotice("Tulis soalnya dulu, atau pilih salah satu contoh.");
      question.focus();
      return;
    }

    showNotice("");
    busy = true;
    solveBtn.disabled = true;
    solveBtn.textContent = "Menghitung…";
    const stopWaiting = startWaiting();

    try {
      showAnswer(await Api.infer(text));
    } catch (err) {
      waiting.hidden = true;
      answerBlock.hidden = true;
      setEmpty(true);
      showNotice(err.message);
    } finally {
      stopWaiting();
      busy = false;
      solveBtn.disabled = false;
      solveBtn.textContent = "Hitung jawaban";
    }
  });
})();
