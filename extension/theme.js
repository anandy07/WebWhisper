// Runs in <head> before paint so a saved theme never flashes the wrong colors.
// data-theme: the user's explicit choice (absent = follow system).
// data-resolved: the theme actually showing, used to pick the toggle icon.
const systemDark = matchMedia("(prefers-color-scheme: dark)");

function applyTheme() {
  let saved = null;
  try {
    saved = localStorage.getItem("theme");
  } catch {}
  const root = document.documentElement;
  if (saved) root.dataset.theme = saved;
  else delete root.dataset.theme;
  root.dataset.resolved = saved || (systemDark.matches ? "dark" : "light");
}

function toggleTheme() {
  const next =
    document.documentElement.dataset.resolved === "dark" ? "light" : "dark";
  try {
    localStorage.setItem("theme", next);
  } catch {}
  applyTheme();
}

applyTheme();
systemDark.addEventListener("change", applyTheme);
