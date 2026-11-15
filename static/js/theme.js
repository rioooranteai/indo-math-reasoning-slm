// Light/dark toggle. The initial theme is set inline in base.html.
(function () {
  const button = document.getElementById("themeToggle");
  const root = document.documentElement;

  function syncLabel() {
    const dark = root.dataset.theme === "dark";
    button.setAttribute("aria-label", dark ? "Ganti ke mode terang" : "Ganti ke mode gelap");
  }

  button.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
    syncLabel();
  });

  syncLabel();
})();
