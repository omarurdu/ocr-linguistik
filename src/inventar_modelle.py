#!/usr/bin/env python3
"""
inventar_modelle.py
Liest die Metadaten der lokal installierten Ollama-Modelle aus und schreibt
  - modelle.md   (Tabelle fuer den Anhang der Hausarbeit)
  - modelle.json (Rohdaten zur Archivierung)

Nur Standardbibliothek. Ollama muss laufen (ollama serve).
Aufruf:  python3 inventar_modelle.py
"""

import json
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

OLLAMA = "http://localhost:11434"
AUSGABE = Path(".")

MODELS = [
    "qwen3-vl:4b",
    "qwen3-vl:8b",
    "gemma4:e4b",
    "qwen3.5:4b",
    "maternion/lightonocr-2:1b",
    "glm-ocr:bf16",
    "deepseek-ocr:3b",
    "minicpm-v4.5:8b",
]

# Felder, die Ollama nicht kennt. Von Hand nachtragen -- das Skript
# erfindet hier nichts, leere Zellen bleiben leer.
MANUELLE_FELDER = ["klasse", "upstream", "referenz", "publikationsstatus", "prompt_id"]


def _request(pfad, payload=None):
    url = OLLAMA + pfad
    daten = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=daten, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as antwort:
        return json.loads(antwort.read().decode())


def ollama_version():
    try:
        return subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "nicht ermittelbar"


def kontextlaenge(model_info):
    """Der Key heisst je nach Architektur anders, z.B. qwen35.context_length."""
    for schluessel, wert in model_info.items():
        if schluessel.endswith(".context_length"):
            return wert
    return ""


def schlank(model_info):
    """Tokenizer-Arrays sind riesig und fuer die Dokumentation wertlos."""
    return {
        k: v
        for k, v in model_info.items()
        if not any(t in k for t in ("merges", "tokenizer.ggml.tokens", "scores", "token_type"))
    }


def erste_zeile(text):
    if not text:
        return ""
    return text.strip().splitlines()[0][:60]


def sammeln():
    tags = {m["model"]: m for m in _request("/api/tags").get("models", [])}
    zeilen, roh, fehlend = [], {}, []

    for name in MODELS:
        eintrag = tags.get(name)
        if eintrag is None:
            fehlend.append(name)
            continue
        try:
            info = _request("/api/show", {"model": name})
        except urllib.error.HTTPError as e:
            fehlend.append(f"{name} (show fehlgeschlagen: {e.code})")
            continue

        details = info.get("details", {})
        mi = info.get("model_info", {})

        zeile = {
            "alias": name,
            "digest_kurz": eintrag.get("digest", "")[:12],
            "digest_voll": eintrag.get("digest", ""),
            "parameter": details.get("parameter_size", ""),
            "quantisierung": details.get("quantization_level", ""),
            "groesse_gb": round(eintrag.get("size", 0) / 1e9, 2),
            "capabilities": ", ".join(info.get("capabilities", [])),
            "kontext": kontextlaenge(mi),
            "familie": details.get("family", ""),
            "format": details.get("format", ""),
            "modified_at": eintrag.get("modified_at", "")[:10],
            "lizenz_hinweis": erste_zeile(info.get("license", "")),
            "defaults": "; ".join(info.get("parameters", "").split("\n")) if info.get("parameters") else "",
        }
        for f in MANUELLE_FELDER:
            zeile[f] = ""

        zeilen.append(zeile)
        roh[name] = {
            "tags": eintrag,
            "details": details,
            "capabilities": info.get("capabilities", []),
            "parameters": info.get("parameters", ""),
            "template": info.get("template", ""),
            "model_info": schlank(mi),
        }

    return zeilen, roh, fehlend


def md_tabelle(kopf, spalten, zeilen):
    aus = ["| " + " | ".join(kopf) + " |",
           "|" + "|".join(["---"] * len(kopf)) + "|"]
    for z in zeilen:
        aus.append("| " + " | ".join(str(z.get(s, "")) for s in spalten) + " |")
    return "\n".join(aus)


def schreiben(zeilen, roh, fehlend):
    heute = datetime.now().strftime("%Y-%m-%d")
    text = [
        "# Modellinventar (lokal, Ollama)",
        "",
        f"- Erhebungsdatum: {heute}",
        f"- Ollama-Version: {ollama_version()}",
        f"- Host: {socket.gethostname()}",
        "- Rohdaten: `modelle.json`",
        "",
        "## A — Automatisch ausgelesen",
        "",
        md_tabelle(
            ["Alias", "Digest", "Parameter", "Quantisierung", "GB", "Capabilities", "Kontext", "Stand"],
            ["alias", "digest_kurz", "parameter", "quantisierung", "groesse_gb",
             "capabilities", "kontext", "modified_at"],
            zeilen,
        ),
        "",
        "## B — Von Hand nachzutragen",
        "",
        md_tabelle(
            ["Alias", "Klasse", "Upstream/Original", "Lizenz", "Referenz", "Publikationsstatus", "Prompt-ID"],
            ["alias", "klasse", "upstream", "lizenz_hinweis", "referenz", "publikationsstatus", "prompt_id"],
            zeilen,
        ),
        "",
        "## C — Laufparameter",
        "",
        "Einheitlich fuer alle lokalen Modelle gesetzt (im Skript, nicht Ollama-Default):",
        "",
        "| Parameter | Wert |",
        "|---|---|",
        "| temperature | 0 |",
        "| seed | <eintragen> |",
        "| top_p | <eintragen> |",
        "| top_k | <eintragen> |",
        "| num_ctx | <eintragen> |",
        "| num_predict | <eintragen> |",
        "| think | <eintragen, siehe unten> |",
        "",
        "Ollama-Defaults je Modell (zur Kontrolle, ob die eigenen Settings greifen):",
        "",
        md_tabelle(["Alias", "Defaults laut /api/show"], ["alias", "defaults"], zeilen),
        "",
        "## D — Bildvorverarbeitung",
        "",
        "| Feld | Wert |",
        "|---|---|",
        "| Quellformat / Farbtiefe | <eintragen> |",
        "| Aufloesung Goldstandard-Ausschnitt (px) | <eintragen> |",
        "| Skalierung vor Uebergabe | <eintragen> |",
        "",
    ]
    if fehlend:
        text += ["## Nicht gefunden", ""] + [f"- `{m}`" for m in fehlend] + [""]

    (AUSGABE / "modelle.md").write_text("\n".join(text), encoding="utf-8")
    (AUSGABE / "modelle.json").write_text(
        json.dumps({"erhebungsdatum": heute, "ollama_version": ollama_version(),
                    "modelle": roh}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        zeilen, roh, fehlend = sammeln()
    except urllib.error.URLError as e:
        sys.exit(f"Ollama nicht erreichbar unter {OLLAMA} ({e}). Laeuft 'ollama serve'?")
    schreiben(zeilen, roh, fehlend)
    print(f"{len(zeilen)} Modelle erfasst -> modelle.md, modelle.json")
    for m in fehlend:
        print(f"  NICHT GEFUNDEN: {m}")