// Shared by the Hitung and Banyak soal pages: API calls, answer comparison
// and rendering of the model's step-by-step solution.

const Api = {
  async infer(question, signal) {
    let response;
    try {
      response = await fetch("/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
        signal,
      });
    } catch (err) {
      if (err.name === "AbortError") throw err;
      throw new Error("Server tidak dapat dihubungi. Pastikan server masih berjalan, lalu coba lagi.");
    }

    if (!response.ok) {
      let detail = "";
      try { detail = (await response.json()).detail || ""; } catch (e) { /* not JSON */ }
      if (response.status === 503) {
        throw new Error("Model masih disiapkan. Tunggu sebentar, lalu coba lagi.");
      }
      throw new Error(detail ? `Model gagal menjawab: ${detail}` : `Model gagal menjawab (kode ${response.status}).`);
    }
    return response.json();
  },
};

const Format = {
  seconds(value) {
    return value.toLocaleString("id-ID", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  },

  // Normalise a numeric answer so "$1,200.00" and "1200" compare equal.
  number(text) {
    if (text == null) return null;
    const cleaned = String(text)
      .replace(/\\text\{[^}]*\}/g, "")
      .replace(/[$€£%\s]|Rp/gi, "")
      .replace(/,(?=\d{3}\b)/g, "");
    const match = cleaned.match(/-?\d+(?:\.\d+)?/);
    if (!match) return null;
    return String(parseFloat(match[0]));
  },

  sameAnswer(a, b) {
    const x = Format.number(a);
    const y = Format.number(b);
    return x !== null && x === y;
  },
};

// Render the model's chain of thought: Markdown plus LaTeX via KaTeX.
const Steps = (() => {
  const SYMBOLS = {
    "\\times": "×", "\\div": "÷", "\\cdot": "·", "\\pm": "±",
    "\\leq": "≤", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈",
    "\\%": "%", "\\$": "$", "\\quad": " ", "\\,": " ",
  };

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function katexHtml(tex, display) {
    return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
  }

  // Split at "=" signs that are not inside braces or brackets.
  function splitAtEquals(tex) {
    const parts = [];
    let depth = 0;
    let start = 0;
    for (let i = 0; i < tex.length; i++) {
      const ch = tex[i];
      if (ch === "{" || ch === "(" || ch === "[") depth++;
      else if (ch === "}" || ch === ")" || ch === "]") depth = Math.max(0, depth - 1);
      else if (ch === "=" && depth === 0 && tex[i - 1] !== "\\") {
        parts.push(tex.slice(start, i));
        start = i + 1;
      }
    }
    parts.push(tex.slice(start));
    return parts.map((part) => part.trim()).filter(Boolean);
  }

  // Display maths, stacked one "= step" per row instead of one long line.
  // Rows use KaTeX inline mode so a row that is still too wide wraps at
  // + and = instead of forcing a horizontal scrollbar.
  function mathBlock(tex) {
    const source = tex.trim();
    if (/\\begin\{/.test(source)) return katexHtml(source, true);

    const parts = splitAtEquals(source);
    const rows = parts.length > 2 || (parts.length === 2 && source.length > 40)
      ? [parts[0], ...parts.slice(1).map((part) => `= ${part}`)]
      : [source];

    const html = rows.map((row, i) =>
      `<span class="math-row${i ? " math-row--next" : ""}">${katexHtml(row, false)}</span>`).join("");
    return `<span class="math-block">${html}</span>`;
  }

  // A bare line such as "3 \times 3 = 9" is maths when little prose is left.
  function looksLikeMath(line) {
    // Pure arithmetic such as "40 + 24 = 64" also counts.
    if (/^[\d\s.,+\-*\/=×÷()%$]+$/.test(line) && line.includes("=") && /\d/.test(line)) return true;
    if (!/\\[a-zA-Z]+|\^|_\{/.test(line)) return false;
    const prose = line
      .replace(/\\text\{[^}]*\}/g, "")
      .replace(/\\[a-zA-Z]+/g, "")
      .replace(/[^A-Za-z]/g, "");
    return prose.length <= 3;
  }

  // Remove the \\boxed{} frame but keep what is inside it.
  function unbox(tex) {
    return tex.replace(/\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}/g, "$1");
  }

  function plainSymbols(text) {
    let out = text.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "$1/$2");
    out = out.replace(/\\text\{([^{}]*)\}/g, "$1");
    for (const [tex, sym] of Object.entries(SYMBOLS)) out = out.split(tex).join(sym);
    return out;
  }

  function render(source) {
    const stash = [];
    const keep = (html) => `MATHSLOT${stash.push(html) - 1}END`;
    const keepBlock = (html) => `MATHBLOCK${stash.push(html) - 1}END`;
    let text = String(source || "").replace(/\r\n/g, "\n");

    // The final answer is already shown large above the steps, so drop the
    // closing "JAWABAN: \\boxed{..}" line (alone or in display maths).
    const BOXED = String.raw`\\boxed\{(?:[^{}]|\{[^{}]*\})*\}`;
    const LABEL = String.raw`(?:\*\*)?(?:jawaban(?: akhir)?|final answer|answer)?(?:\*\*)?\s*:?\s*(?:\*\*)?`;
    text = text.replace(new RegExp(String.raw`^[ \t]*${LABEL}[ \t]*(?:\\\[|\$\$)?\s*${BOXED}\s*(?:\\\]|\$\$)?[ \t]*\.?[ \t]*$`, "gim"), "");
    text = text.replace(/\n{3,}/g, "\n\n").trim();

    // Delimited maths first, so Markdown never sees the backslashes.
    text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => keepBlock(mathBlock(unbox(tex))));
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => keepBlock(mathBlock(unbox(tex))));
    text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => keep(katexHtml(unbox(tex), false)));

    // A \\boxed{..} left inside a sentence keeps its value, without the frame.
    text = text.replace(/\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}/g, (_, tex) => keep(katexHtml(tex, false)));

    text = text.split("\n").map((line) => {
      const trimmed = line.trim();
      if (!trimmed || /MATH(SLOT|BLOCK)/.test(trimmed)) return line;
      if (looksLikeMath(trimmed)) {
        // Keep the indentation so the line stays inside its numbered step.
        const indent = line.match(/^\s*/)[0];
        const tex = trimmed.replace(/\*/g, "\\times ").replace(/\$/g, "\\$");
        return indent + keepBlock(mathBlock(tex));
      }
      return plainSymbols(line);
    }).join("\n");

    let html = window.marked.parse(escapeHtml(text), { breaks: true, gfm: true });
    // A calculation is its own block, so drop the line breaks around it.
    html = html
      .replace(/<br>\s*(?=MATHBLOCK)/g, "")
      .replace(/(MATHBLOCK\d+END)\s*<br>/g, "$1")
      .replace(/MATH(?:SLOT|BLOCK)(\d+)END/g, (_, i) => stash[Number(i)]);
    return html;
  }

  return { render };
})();
