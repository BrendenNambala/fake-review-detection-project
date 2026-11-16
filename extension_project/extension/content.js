/*
 * EXPERIMENTAL: badges reviews directly on a Google Maps place page.
 *
 * Honest warning: Google Maps uses scrambled, frequently-changing CSS class names,
 * so the selectors below can break at any time. This is a proof of concept for the
 * demo, not a robust product. If it finds nothing, it does nothing (the popup still
 * works). It only reads review text already displayed in YOUR browser and sends it
 * to YOUR own API; it does not call Google's servers or scrape in bulk.
 */
const DEFAULT_API = "http://127.0.0.1:8000/predict";

async function getApiUrl() {
  const { apiUrl } = await chrome.storage.local.get("apiUrl");
  return apiUrl || DEFAULT_API;
}

// Candidate selectors for a single review's text span (several fallbacks).
const REVIEW_SELECTORS = [
  ".wiI7pd",          // common review text span
  ".MyEned",          // review container variant
  "[class*='review-full-text']",
];

function findReviewNodes() {
  for (const sel of REVIEW_SELECTORS) {
    const nodes = Array.from(document.querySelectorAll(sel))
      .filter((n) => n.textContent && n.textContent.trim().length > 15);
    if (nodes.length) return nodes;
  }
  return [];
}

function makeBadge(text) {
  const b = document.createElement("span");
  b.className = "frd-badge frd-loading";
  b.textContent = "checking…";
  return b;
}

async function classify(api, text) {
  try {
    const res = await fetch(api, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    return await res.json();
  } catch (e) {
    return null;
  }
}

async function run() {
  const api = await getApiUrl();
  const nodes = findReviewNodes();
  for (const node of nodes) {
    if (node.dataset.frdDone) continue;      // avoid re-processing
    node.dataset.frdDone = "1";
    const text = node.textContent.trim();
    const badge = makeBadge(text);
    node.parentElement && node.parentElement.insertBefore(badge, node);
    const data = await classify(api, text);
    if (!data || data.error) {
      badge.className = "frd-badge frd-unknown";
      badge.textContent = "FRD: n/a";
      continue;
    }
    const pct = Math.round((data.fake_probability || 0) * 100);
    const fake = data.verdict === "LIKELY FAKE";
    badge.className = "frd-badge " + (fake ? "frd-fake" : "frd-genuine");
    badge.textContent = (fake ? "⚠ Likely fake " : "✔ Likely genuine ") + pct + "%";
  }
}

// Google Maps loads reviews lazily as you scroll, so re-check periodically.
let ticks = 0;
const timer = setInterval(() => {
  run();
  if (++ticks > 40) clearInterval(timer); // stop after ~2 minutes
}, 3000);
run();
