// Service worker: adds a right-click menu to check any selected review text.
const DEFAULT_API = "http://127.0.0.1:8000/predict";

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.local.get("apiUrl");
  return apiUrl || DEFAULT_API;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkReview",
    title: "Check this review for fakeness",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info) => {
  if (info.menuItemId !== "checkReview" || !info.selectionText) return;
  try {
    const api = await getApiUrl();
    const res = await fetch(api, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: info.selectionText }),
    });
    const data = await res.json();
    const pct = Math.round((data.fake_probability || 0) * 100);
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon128.png",
      title: data.verdict || "Fake Review Detector",
      message: (data.verdict === "LIKELY FAKE" ? "⚠ " : "✔ ") + "Fake probability: " + pct + "%",
    });
  } catch (e) {
    chrome.notifications.create({
      type: "basic",
      iconUrl: "icon128.png",
      title: "Fake Review Detector",
      message: "Could not reach the API. Check the API URL and that it is running.",
    });
  }
});
