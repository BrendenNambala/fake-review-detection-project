import re
import os
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

# Render crisply on high-DPI Windows displays (fixes blurry text).
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        windll.user32.SetProcessDPIAware()
    except Exception:
        pass

SIDEBAR = "#0f172a"; SIDEBAR_ACT = "#1e293b"; ACCENT = "#2563eb"
BG = "#f1f5f9"; CARD = "#ffffff"; LINE = "#e2e8f0"; TEXT = "#0f172a"; MUTED = "#64748b"
GREEN = "#15803d"; GREEN_BG = "#dcfce7"; RED = "#b91c1c"; RED_BG = "#fee2e2"

# NOTE: keep this list in sync with the slang list used to TRAIN the model.
# After saving this file, re-run your slang-sync cell to restore your full (19-word) list.
SLANG = ['lekker', 'sharp sharp', 'sharp', 'the plug', 'no cap', 'hey', 'proper', 'bru', 'yoh', 'eish', 'giving', 'clean clean', 'boss', 'nxa', 'mooi', 'best best', 'shuu', 'shomafa', 'shiwa nee']

HISTORY_FILE = "analysis_history.csv"

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    def sentiment(t):
        return _vader.polarity_scores(t)["compound"]
except Exception:
    def sentiment(t):
        return 0.0

def clean_text(t):
    t = t.lower(); t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s!]", " ", t); return re.sub(r"\s+", " ", t).strip()

def eng_features(text):
    low, cc = text.lower(), len(text)
    return np.array([[len(text.split()), cc, text.count("!"),
        (sum(1 for c in text if c.isupper()) / cc if cc else 0),
        sum(low.count(s) for s in SLANG), sentiment(text)]], dtype=float)

try:
    model = joblib.load("demo_model.pkl")
    vec = joblib.load("demo_vectorizer.pkl")
    scaler = joblib.load("demo_scaler.pkl")
    MODEL_OK = True
    try:
        FEAT_NAMES = list(vec.get_feature_names_out())
    except Exception:
        FEAT_NAMES = list(vec.get_feature_names())
    N_VOCAB = len(FEAT_NAMES)
except Exception as e:
    MODEL_OK = False; MODEL_ERR = str(e)

def classify_many(texts):
    clean = [clean_text(t) for t in texts]
    X = vec.transform(clean)
    F = scaler.transform(np.vstack([eng_features(t)[0] for t in texts]))
    return model.predict_proba(hstack([X, csr_matrix(F)]))[:, 1]

def top_words(text, k=8):
    row = vec.transform([clean_text(text)]).tocoo()
    coef = model.coef_[0]
    contribs = [(FEAT_NAMES[i], v * coef[i]) for i, v in zip(row.col, row.data) if i < N_VOCAB]
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:k]

# ---------- history helpers ----------
def log_analysis(text, verdict, prob):
    """Append one analysis to the history CSV."""
    rec = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review": text.replace("\n", " ").strip(),
        "verdict": verdict,
        "fake_probability": round(float(prob), 3),
    }
    try:
        header = not os.path.exists(HISTORY_FILE)
        pd.DataFrame([rec]).to_csv(HISTORY_FILE, mode="a", header=header, index=False)
    except Exception:
        pass  # never let logging break the app

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE)
    except Exception:
        pass
    return pd.DataFrame(columns=["timestamp", "review", "verdict", "fake_probability"])

root = tk.Tk()
root.title("Fake Review Detector")
root.geometry("1040x680"); root.configure(bg=BG); root.minsize(940, 620)

style = ttk.Style()
try: style.theme_use("clam")
except Exception: pass
style.configure("Modern.Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT,
                rowheight=28, font=("Segoe UI", 10), borderwidth=0)
style.configure("Modern.Treeview.Heading", background="#e2e8f0", foreground=TEXT,
                font=("Segoe UI", 10, "bold"), relief="flat", padding=6)
style.map("Modern.Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", TEXT)])

root.grid_columnconfigure(1, weight=1); root.grid_rowconfigure(0, weight=1)

side = tk.Frame(root, bg=SIDEBAR, width=220); side.grid(row=0, column=0, sticky="nsew"); side.grid_propagate(False)
tk.Label(side, text="  Fake Review\n  Detector", font=("Segoe UI", 15, "bold"), fg="white",
         bg=SIDEBAR, justify="left").pack(anchor="w", pady=(24, 2), padx=6)
tk.Label(side, text="  Namibian consumer reviews", font=("Segoe UI", 8), fg="#94a3b8",
         bg=SIDEBAR).pack(anchor="w", padx=6, pady=(0, 24))

navbtns = {}
def show(key):
    pages[key].tkraise()
    if key == "history":
        refresh_history()
    for k, b in navbtns.items():
        b.config(bg=SIDEBAR_ACT if k == key else SIDEBAR, fg="white" if k == key else "#cbd5e1")

def make_nav(label, key):
    b = tk.Button(side, text="   " + label, font=("Segoe UI", 11), fg="#cbd5e1", bg=SIDEBAR,
                  activebackground=SIDEBAR_ACT, activeforeground="white", relief="flat",
                  anchor="w", padx=16, pady=12, bd=0, command=lambda: show(key))
    b.pack(fill="x"); navbtns[key] = b

for lab, key in [("Detector", "detector"), ("Batch Scan", "batch"), ("Dataset", "dataset"),
                 ("Model Results", "results"), ("Feature Importance", "features"), ("History", "history")]:
    make_nav(lab, key)

content = tk.Frame(root, bg=BG); content.grid(row=0, column=1, sticky="nsew")
content.grid_rowconfigure(0, weight=1); content.grid_columnconfigure(0, weight=1)
pages = {}
def new_page():
    p = tk.Frame(content, bg=BG); p.grid(row=0, column=0, sticky="nsew"); return p
def card(parent, **kw):
    return tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1, **kw)
def stat_card(parent, value, label, accent=ACCENT):
    c = card(parent)
    tk.Label(c, text=value, font=("Segoe UI", 22, "bold"), fg=accent, bg=CARD).pack(anchor="w", padx=16, pady=(14, 0))
    tk.Label(c, text=label, font=("Segoe UI", 9), fg=MUTED, bg=CARD).pack(anchor="w", padx=16, pady=(0, 14))
    return c
def page_title(parent, title, sub):
    tk.Label(parent, text=title, font=("Segoe UI", 18, "bold"), fg=TEXT, bg=BG).pack(anchor="w", padx=28, pady=(22, 0))
    tk.Label(parent, text=sub, font=("Segoe UI", 10), fg=MUTED, bg=BG).pack(anchor="w", padx=28, pady=(0, 14))

# ---------- DETECTOR ----------
p1 = new_page(); pages["detector"] = p1
page_title(p1, "Review Detector", "Classify a single review and see the evidence behind the verdict.")
dc = card(p1); dc.pack(fill="both", expand=True, padx=28, pady=(0, 22))
tk.Label(dc, text="Review text", font=("Segoe UI", 10, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=18, pady=(16, 6))
input_box = scrolledtext.ScrolledText(dc, height=4, font=("Segoe UI", 11), wrap="word", relief="solid", borderwidth=1)
input_box.pack(fill="x", padx=18)

exrow = tk.Frame(dc, bg=CARD); exrow.pack(anchor="w", padx=18, pady=(8, 0))
def fill_example(t):
    input_box.delete("1.0", "end"); input_box.insert("1.0", t)
tk.Label(exrow, text="Try:", font=("Segoe UI", 9), fg=MUTED, bg=CARD).pack(side="left")
for lbl, txt in [("hype example", "BEST SHOP EVER!!! amazing prices, highly recommend to everyone!!!"),
                 ("genuine example", "I bought a charger here last week, it works fine and they gave me a receipt.")]:
    tk.Button(exrow, text=lbl, font=("Segoe UI", 8), fg=ACCENT, bg="#eff6ff", relief="flat",
              padx=8, pady=2, cursor="hand2", command=lambda t=txt: fill_example(t)).pack(side="left", padx=4)

verdict_fr = tk.Frame(dc, bg=CARD); verdict_fr.pack(fill="x", padx=18, pady=(14, 6))
verdict_lbl = tk.Label(verdict_fr, text="", font=("Segoe UI", 14, "bold"), bg=CARD, anchor="w", padx=16, pady=12)
feat_lbl = tk.Label(dc, text="", font=("Consolas", 10), fg="#334155", bg=CARD, justify="left")
feat_lbl.pack(anchor="w", padx=18)
ev_title = tk.Label(dc, text="", font=("Segoe UI", 10, "bold"), fg=TEXT, bg=CARD)
ev_title.pack(anchor="w", padx=18, pady=(10, 2))
ev_fr = tk.Frame(dc, bg=CARD); ev_fr.pack(anchor="w", padx=16, pady=(0, 10), fill="x")

def analyse():
    if not MODEL_OK:
        messagebox.showerror("Model not loaded", "Could not load model files:\n" + MODEL_ERR); return
    text = input_box.get("1.0", "end").strip()
    if not text: return
    prob = float(classify_many([text])[0])
    verdict = "LIKELY FAKE" if prob >= 0.5 else "LIKELY GENUINE"
    verdict_lbl.pack(fill="x")
    if prob >= 0.5:
        verdict_fr.config(bg=RED_BG); verdict_lbl.config(text=f"⚠  LIKELY FAKE        fake probability  {prob:.0%}", bg=RED_BG, fg=RED)
    else:
        verdict_fr.config(bg=GREEN_BG); verdict_lbl.config(text=f"✔  LIKELY GENUINE        fake probability  {prob:.0%}", bg=GREEN_BG, fg=GREEN)
    fe = eng_features(text)[0]
    feat_lbl.config(text=(f"words {int(fe[0])}    characters {int(fe[1])}    exclamations {int(fe[2])}    "
                          f"capitals {fe[3]:.2f}    slang {int(fe[4])}    sentiment {fe[5]:.2f}"))
    for w in ev_fr.winfo_children(): w.destroy()
    words = top_words(text)
    if words:
        ev_title.config(text="Words that drove this verdict:")
        for i, (term, contrib) in enumerate(words):
            if contrib >= 0:
                bgc, fgc, mark = RED_BG, RED, "↑fake"
            else:
                bgc, fgc, mark = GREEN_BG, GREEN, "↓genuine"
            chip = tk.Label(ev_fr, text=f" {term}  ({mark}) ", font=("Segoe UI", 9), bg=bgc, fg=fgc, padx=6, pady=3)
            chip.grid(row=i // 4, column=i % 4, padx=4, pady=4, sticky="w")
    else:
        ev_title.config(text="No known words from this review appeared in the model's vocabulary.")
    log_analysis(text, verdict, prob)  # record this analysis

tk.Button(dc, text="Analyse review", font=("Segoe UI", 11, "bold"), fg="white", bg=ACCENT,
          activebackground="#1e4fd0", activeforeground="white", relief="flat", padx=20, pady=9,
          cursor="hand2", command=analyse).pack(anchor="w", padx=18, pady=(12, 16))

# ---------- BATCH ----------
p2 = new_page(); pages["batch"] = p2
page_title(p2, "Batch Scan", "Load a CSV of many reviews and classify them all at once.")
bbar = tk.Frame(p2, bg=BG); bbar.pack(fill="x", padx=28)
batch_info = tk.StringVar(value="No file loaded.")
batch_state = {"df": None}

def load_batch():
    path = filedialog.askopenfilename(title="Choose a CSV of reviews", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not path: return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Could not read file", str(e)); return
    textcol = None
    for cand in ["text", "review", "Review", "Text", "content"]:
        if cand in df.columns: textcol = cand; break
    if textcol is None:
        obj = [c for c in df.columns if df[c].dtype == object]
        if not obj:
            messagebox.showerror("No text column", "Could not find a column of review text in that file."); return
        textcol = obj[0]
    texts = df[textcol].astype(str).tolist()
    probs = classify_many(texts)
    out = pd.DataFrame({"review": texts, "fake_probability": np.round(probs, 3),
                        "verdict": ["LIKELY FAKE" if p >= 0.5 else "LIKELY GENUINE" for p in probs]})
    batch_state["df"] = out
    for r in tv.get_children(): tv.delete(r)
    for n, row in out.iterrows():
        short = row["review"][:70] + ("..." if len(row["review"]) > 70 else "")
        tv.insert("", "end", values=(short, row["verdict"], f"{row['fake_probability']:.0%}"),
                  tags=("fake" if row["verdict"] == "LIKELY FAKE" else "genuine",))
    nfake = int((out["verdict"] == "LIKELY FAKE").sum())
    batch_info.set(f"{len(out)} reviews scanned   —   {nfake} flagged as likely fake,  {len(out) - nfake} genuine")

def export_batch():
    if batch_state["df"] is None:
        messagebox.showinfo("Nothing to export", "Load and scan a CSV first."); return
    path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="batch_results.csv",
                                        filetypes=[("CSV files", "*.csv")])
    if path:
        batch_state["df"].to_csv(path, index=False)
        messagebox.showinfo("Saved", "Results saved to:\n" + path)

tk.Button(bbar, text="Load CSV", font=("Segoe UI", 10, "bold"), fg="white", bg=ACCENT, relief="flat",
          padx=16, pady=7, cursor="hand2", command=load_batch).pack(side="left")
tk.Button(bbar, text="Export results", font=("Segoe UI", 10), fg=TEXT, bg="#e2e8f0", relief="flat",
          padx=16, pady=7, cursor="hand2", command=export_batch).pack(side="left", padx=8)
tk.Label(p2, textvariable=batch_info, font=("Segoe UI", 10), fg=MUTED, bg=BG).pack(anchor="w", padx=28, pady=(10, 6))
btc = card(p2); btc.pack(fill="both", expand=True, padx=28, pady=(0, 22))
tv = ttk.Treeview(btc, columns=("review", "verdict", "prob"), show="headings", style="Modern.Treeview")
tv.heading("review", text="Review"); tv.column("review", width=480, anchor="w")
tv.heading("verdict", text="Verdict"); tv.column("verdict", width=140, anchor="center")
tv.heading("prob", text="Fake %"); tv.column("prob", width=90, anchor="center")
tv.tag_configure("fake", foreground=RED); tv.tag_configure("genuine", foreground=GREEN)
sb = ttk.Scrollbar(btc, orient="vertical", command=tv.yview); tv.configure(yscrollcommand=sb.set)
tv.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10); sb.pack(side="right", fill="y", pady=10)

# ---------- DATASET ----------
p3 = new_page(); pages["dataset"] = p3
page_title(p3, "Dataset", "The labelled Namibian review dataset built for this study.")
try:
    df = pd.read_csv("namibian_reviews_final.csv")
    total = len(df); fl = pd.to_numeric(df["final_label"], errors="coerce")
    genuine = int((fl == 0).sum()); fake = int((fl == 1).sum()); kappa_txt = "n/a"
    if "label_annotator1" in df.columns and "label_annotator2" in df.columns:
        try:
            from sklearn.metrics import cohen_kappa_score
            a1 = pd.to_numeric(df["label_annotator1"], errors="coerce")
            a2 = pd.to_numeric(df["label_annotator2"], errors="coerce")
            m = a1.notna() & a2.notna()
            kappa_txt = f"{cohen_kappa_score(a1[m].astype(int), a2[m].astype(int)):.3f}"
        except Exception: pass
    row = tk.Frame(p3, bg=BG); row.pack(fill="x", padx=22)
    for i, (v, l, col) in enumerate([(str(total), "Total reviews", ACCENT), (str(genuine), "Genuine", GREEN),
                                     (str(fake), "Fake", RED), (kappa_txt, "Cohen's Kappa", "#7c3aed")]):
        stat_card(row, v, l, col).grid(row=0, column=i, padx=6, sticky="ew"); row.grid_columnconfigure(i, weight=1)
    cols = tk.Frame(p3, bg=BG); cols.pack(fill="both", expand=True, padx=22, pady=16)
    cols.grid_columnconfigure(0, weight=1); cols.grid_columnconfigure(1, weight=1)
    if "product_category" in df.columns:
        c1 = card(cols); c1.grid(row=0, column=0, padx=6, sticky="nsew")
        tk.Label(c1, text="By business category", font=("Segoe UI", 11, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=16, pady=(14, 6))
        for cat, n in df["product_category"].value_counts().items():
            tk.Label(c1, text=f"{cat}:  {n}", font=("Segoe UI", 10), fg="#334155", bg=CARD).pack(anchor="w", padx=16, pady=1)
        tk.Label(c1, text="", bg=CARD).pack(pady=4)
    if "source" in df.columns:
        c2 = card(cols); c2.grid(row=0, column=1, padx=6, sticky="nsew")
        tk.Label(c2, text="By source", font=("Segoe UI", 11, "bold"), fg=TEXT, bg=CARD).pack(anchor="w", padx=16, pady=(14, 6))
        for src, n in df["source"].value_counts().items():
            tk.Label(c2, text=f"{src}:  {n}", font=("Segoe UI", 10), fg="#334155", bg=CARD).pack(anchor="w", padx=16, pady=1)
        tk.Label(c2, text="Two annotators labelled independently;\nKappa measures their agreement.",
                 font=("Segoe UI", 9), fg=MUTED, bg=CARD, justify="left").pack(anchor="w", padx=16, pady=(10, 14))
except Exception as e:
    tk.Label(p3, text="Could not load namibian_reviews_final.csv\n" + str(e), font=("Segoe UI", 10), fg=RED, bg=BG, justify="left").pack(anchor="w", padx=28)

# ---------- RESULTS ----------
p4 = new_page(); pages["results"] = p4
page_title(p4, "Model Results", "Five-fold cross-validated performance on the fake class.")
try:
    dfr = pd.read_csv("results_model_comparison.csv")
    tc = card(p4); tc.pack(fill="x", padx=28, pady=(0, 10))
    cc = list(dfr.columns)
    tv2 = ttk.Treeview(tc, columns=cc, show="headings", height=len(dfr), style="Modern.Treeview")
    for c in cc:
        tv2.heading(c, text=c.replace("_", " ").title()); tv2.column(c, width=150, anchor="center")
    for n, (_, r) in enumerate(dfr.iterrows()):
        tv2.insert("", "end", values=[r[c] for c in cc], tags=("odd" if n % 2 else "even",))
    tv2.tag_configure("odd", background="#f8fafc"); tv2.tag_configure("even", background=CARD)
    tv2.pack(fill="x", padx=10, pady=10)
    cal = tk.Frame(p4, bg="#eff6ff", highlightbackground="#bfdbfe", highlightthickness=1); cal.pack(fill="x", padx=28, pady=6)
    tk.Label(cal, text="Best model:  Logistic Regression  —  highest F1 on the fake class",
             font=("Segoe UI", 10, "bold"), fg="#1e40af", bg="#eff6ff").pack(anchor="w", padx=14, pady=10)
except Exception as e:
    tk.Label(p4, text="Could not load results_model_comparison.csv\n" + str(e), font=("Segoe UI", 10), fg=RED, bg=BG, justify="left").pack(anchor="w", padx=28)

# ---------- FEATURES ----------
p5 = new_page(); pages["features"] = p5
page_title(p5, "Feature Importance", "What the model learned  —  red points to fake, green points to genuine.")
try:
    dff = pd.read_csv("results_feature_importance.csv")
    fc = card(p5); fc.pack(fill="both", expand=True, padx=28, pady=(0, 22))
    cv = tk.Canvas(fc, bg=CARD, highlightthickness=0); cv.pack(fill="both", expand=True, padx=14, pady=14)
    coefs = pd.to_numeric(dff["coefficient"], errors="coerce").fillna(0)
    maxabs = max(abs(coefs.min()), abs(coefs.max()), 0.001); base = 190; y = 24
    for _, r in dff.iterrows():
        name = str(r["feature"]); c = float(r["coefficient"]); length = int(abs(c) / maxabs * 230)
        colour = "#dc2626" if c >= 0 else "#16a34a"; tag = "FAKE" if c >= 0 else "GENUINE"
        cv.create_text(24, y + 11, text=name, anchor="w", font=("Segoe UI", 11))
        cv.create_rectangle(base, y, base + length, y + 22, fill=colour, outline="")
        cv.create_text(base + length + 10, y + 11, text=f"{c:+.2f}  {tag}", anchor="w", font=("Consolas", 9), fill="#334155")
        y += 42
except Exception as e:
    tk.Label(p5, text="Could not load results_feature_importance.csv\n" + str(e), font=("Segoe UI", 10), fg=RED, bg=BG, justify="left").pack(anchor="w", padx=28)

# ---------- HISTORY ----------
p6 = new_page(); pages["history"] = p6
page_title(p6, "History", "A record of every review analysed in the Detector.")
hbar = tk.Frame(p6, bg=BG); hbar.pack(fill="x", padx=28)
history_info = tk.StringVar(value="")

def refresh_history():
    h = load_history()
    for r in htv.get_children():
        htv.delete(r)
    if len(h):
        for _, row in h.iloc[::-1].iterrows():  # newest first
            rev = str(row.get("review", ""))
            short = rev[:70] + ("..." if len(rev) > 70 else "")
            try:
                prob = float(row.get("fake_probability", 0))
                probtxt = f"{prob:.0%}"
            except Exception:
                probtxt = str(row.get("fake_probability", ""))
            verdict = str(row.get("verdict", ""))
            htv.insert("", "end",
                       values=(row.get("timestamp", ""), short, verdict, probtxt),
                       tags=("fake" if verdict == "LIKELY FAKE" else "genuine",))
    history_info.set(f"{len(h)} analyses recorded.")

def clear_history():
    if not messagebox.askyesno("Clear history", "Delete all recorded analyses? This cannot be undone."):
        return
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        messagebox.showerror("Could not clear", str(e)); return
    refresh_history()

def export_history():
    h = load_history()
    if not len(h):
        messagebox.showinfo("Nothing to export", "No analyses recorded yet."); return
    path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="analysis_history.csv",
                                        filetypes=[("CSV files", "*.csv")])
    if path:
        h.to_csv(path, index=False)
        messagebox.showinfo("Saved", "History saved to:\n" + path)

tk.Button(hbar, text="Refresh", font=("Segoe UI", 10, "bold"), fg="white", bg=ACCENT, relief="flat",
          padx=16, pady=7, cursor="hand2", command=refresh_history).pack(side="left")
tk.Button(hbar, text="Export", font=("Segoe UI", 10), fg=TEXT, bg="#e2e8f0", relief="flat",
          padx=16, pady=7, cursor="hand2", command=export_history).pack(side="left", padx=8)
tk.Button(hbar, text="Clear history", font=("Segoe UI", 10), fg=RED, bg="#fee2e2", relief="flat",
          padx=16, pady=7, cursor="hand2", command=clear_history).pack(side="left")
tk.Label(p6, textvariable=history_info, font=("Segoe UI", 10), fg=MUTED, bg=BG).pack(anchor="w", padx=28, pady=(10, 6))
hc = card(p6); hc.pack(fill="both", expand=True, padx=28, pady=(0, 22))
htv = ttk.Treeview(hc, columns=("time", "review", "verdict", "prob"), show="headings", style="Modern.Treeview")
htv.heading("time", text="Time"); htv.column("time", width=150, anchor="w")
htv.heading("review", text="Review"); htv.column("review", width=420, anchor="w")
htv.heading("verdict", text="Verdict"); htv.column("verdict", width=140, anchor="center")
htv.heading("prob", text="Fake %"); htv.column("prob", width=80, anchor="center")
htv.tag_configure("fake", foreground=RED); htv.tag_configure("genuine", foreground=GREEN)
hsb = ttk.Scrollbar(hc, orient="vertical", command=htv.yview); htv.configure(yscrollcommand=hsb.set)
htv.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10); hsb.pack(side="right", fill="y", pady=10)
refresh_history()

show("detector")
root.mainloop()
