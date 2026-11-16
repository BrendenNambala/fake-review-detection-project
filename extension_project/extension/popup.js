// Default API URL. Change it in "API settings" after you deploy your API,
// or set it to http://127.0.0.1:8000/predict while testing locally.
const DEFAULT_API = "http://127.0.0.1:8000/predict";

const $ = (id) => document.getElementById(id);

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.local.get("apiUrl");
  return apiUrl || DEFAULT_API;
}

// Prefill saved URL
getApiUrl().then((u) => { $("apiUrl").value = u; });

$("saveUrl").addEventListener("click", async () => {
  const u = $("apiUrl").value.trim();
  if (u) { await chrome.storage.local.set({ apiUrl: u }); $("status").textContent = "API URL saved."; }
});

// Grab whatever the user has highlighted on the current page
$("useSelection").addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    if (result && result.trim()) { $("review").value = result.trim(); $("status").textContent = ""; }
    else { $("status").textContent = "No text selected on the page."; }
  } catch (e) {
    $("status").textContent = "Could not read the page selection.";
  }
});

$("analyse").addEventListener("click", async () => {
  const text = $("review").value.trim();
  const verdict = $("verdict");
  const evidence = $("evidence");
  verdict.style.display = "none";
  evidence.innerHTML = "";
  if (!text) { $("status").textContent = "Enter or select a review first."; return; }
  $("status").textContent = "Analysing...";
  try {
    const api = await getApiUrl();
    const res = await fetch(api, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (data.error) { $("status").textContent = "API error: " + data.error; return; }
    const pct = Math.round(data.fake_probability * 100);
    verdict.className = data.verdict === "LIKELY FAKE" ? "fake" : "genuine";
    verdict.textContent = (data.verdict === "LIKELY FAKE" ? "⚠ " : "✔ ") + data.verdict + "  —  fake probability " + pct + "%";
    verdict.style.display = "block";
    (data.evidence || []).forEach((e) => {
      const c = document.createElement("span");
      c.className = "chip " + (e.direction === "fake" ? "f" : "g");
      c.textContent = e.word + (e.direction === "fake" ? " ↑fake" : " ↓genuine");
      evidence.appendChild(c);
    });
    $("status").textContent = "";
  } catch (e) {
    $("status").textContent = "Could not reach the API. Check the API URL in settings and that the API is running.";
  }
});
