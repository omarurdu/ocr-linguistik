"""
frontier_ocr.py - Gegenstueck zu ollama_ocr.py fuer die API-Modelle.

Aufbau bewusst parallel zu ollama_ocr.py: gleicher Prompt (P-GEN-01), gleiche
Verzeichnisse, gleiche Kernspalten im Ergebnis-CSV. Die beiden CSVs lassen sich
daher mit pd.concat zusammenfuehren.

Zentrale Designentscheidung: PARAMS ist nach MODELL geschluesselt, nicht nach
Anbieter. Grund ist ein empirischer Befund aus dem ersten Lauf - die Parameter-
Oberflaeche haengt an der Modellgeneration, nicht an der Firma. Claude Opus 5
akzeptiert kein temperature und denkt per Default; Claude Haiku 4.5 akzeptiert
temperature, aber kein effort. Der Aufruf wird ausschliesslich aus diesem Dict
gebaut. Damit kann params_json nicht von dem abweichen, was gesendet wurde.

Benoetigt:  pip install anthropic openai google-genai python-dotenv pandas
"""

import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Pfade - identisch zu ollama_ocr.py (Skript liegt in src/)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "transcriptions"
RAW_DIR = OUTPUT_DIR / "raw"
# Sammelordner fuer alle Tabellen-Ausgaben des Projekts (CSV/MD/JSON).
TABELLEN_DIR = OUTPUT_DIR / "tabellen"

for _d in (TRANSCRIPTIONS_DIR, RAW_DIR, TABELLEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# Projektwurzel auf den Modul-Suchpfad, damit prompts/ importierbar ist -
# unabhaengig davon, aus welchem Verzeichnis das Skript gestartet wird.
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.prompts_base import PROMPT_ID_GEN, PROMPT_TEXTS  # noqa: E402

# ---------------------------------------------------------------------------
# Testlauf: erst einmal nur anzeige01.png. Zum Ausweiten IMAGE_FILTER auf None
# setzen - dann werden alle PNGs aus data/images/ verarbeitet.
# ---------------------------------------------------------------------------

IMAGE_FILTER = None

# Vorbereitung fuer die Varianzpruefung. Bleibt vorerst 1; bei N_RUNS > 1
# bekommen die Dateinamen ein __runN-Suffix, run_index landet im CSV.
N_RUNS = 1

# ---------------------------------------------------------------------------
# Modelle. ABRUFDATUM DER MODELL-IDS: 2026-08-12
#
# Opus 5 vs. Sonnet 5 statt Opus 5 vs. Haiku 4.5: gleiche Modellgeneration,
# gleiche Parameter-Oberflaeche. Der Kontrast misst dann das Tier und nicht
# zusaetzlich den Generationswechsel.
# ---------------------------------------------------------------------------

MODELS = [
    {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "role": "Frontier-Obergrenze A",
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "role": "Tier-Kontrast zu A, gleiche Generation",
    },
    {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "role": "Frontier-Obergrenze B",
    },
    {
        "provider": "google",
        "model": "gemini-3.5-flash",
        "role": "Frontier-Obergrenze C (stable, nicht preview)",
    },
]

# ---------------------------------------------------------------------------
# Prompt: kommt aus prompts/prompts_base.py, derselbe Eintrag, den auch
# ollama_ocr.py fuer die Generalisten zieht. Der Text steht nur dort - sobald er
# zwischen den Skripten abweicht, misst der Vergleich Prompt und Modell zugleich.
# Herkunft des Prompts: PROMPT_META in prompts_base.py.
# ---------------------------------------------------------------------------

PROMPT_ID = PROMPT_ID_GEN
PROMPT_GEN = PROMPT_TEXTS[PROMPT_ID]

# ---------------------------------------------------------------------------
# Parameter je MODELL. Die Asymmetrie ist Befund, nicht Schlamperei, und gehoert
# so in den Methodenteil:
#
#   claude-opus-5 / claude-sonnet-5
#       temperature ab Claude 4.7 deprecated -> wird nicht gesendet (400er).
#       Thinking ist auf Claude-5-Modellen per Default AN und wird hier
#       explizit abgeschaltet -> Pendant zu think=False in ollama_ocr.py.
#       effort wird auf den API-Default "high" gesetzt, rein zur Dokumentation.
#       (Auf Haiku 4.5 waere effort NICHT unterstuetzt - deshalb ist dieses
#       Dict nach Modell und nicht nach Anbieter geschluesselt.)
#
#   gpt-5.6-sol
#       Die GPT-5-Reasoning-Linie nimmt kein temperature; Steuerung ueber
#       reasoning effort. "none" ist das Pendant zu think=False.
#       [pruefen: akzeptiert das Konto effort="none"? sonst "low" und vermerken]
#
#   gemini-3.5-flash
#       temperature/top_p/top_k sind in der Gemini-API deprecated und werden
#       gar nicht erst gesendet. Gemini 3.x denkt per Default dynamisch; eine
#       saubere Instruct-Variante gibt es nicht. Das ist die verbleibende
#       Asymmetrie im Design und der Grund, warum die Varianzpruefung bei
#       allen API-Modellen Pflicht ist.
#
# max_tokens grosszuegiger als num_predict=1024 in ollama_ocr.py: eine hohe
# Obergrenze kann die Transkriptionsguete nicht verbessern, verhindert aber
# kuenstliche Abschnitte.
# ---------------------------------------------------------------------------

PARAMS = {
    "claude-opus-5": {
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "high"},
    },
    "claude-sonnet-5": {
        "max_tokens": 4096,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "high"},
    },
    "gpt-5.6-sol": {
        "max_output_tokens": 4096,
        "reasoning_effort": "none",
    },
    "gemini-3.5-flash": {
        "max_output_tokens": 4096,
    },
}

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 4  # Sekunden, verdoppelt sich je Versuch


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_jsonable(obj):
    """SDK-Antwortobjekte in ein serialisierbares dict ueberfuehren."""
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return {"repr": repr(obj)}


def normalize_done_reason(raw) -> str:
    """
    Auf das Vokabular von ollama_ocr.py abbilden, damit die Abschneide-Pruefung
    (done_reason == 'length') ueber beide Skripte hinweg funktioniert.
    """
    if raw is None:
        return None
    r = str(raw).lower()
    if r in {"max_tokens", "max_output_tokens", "length"}:
        return "length"
    if r in {"end_turn", "stop", "completed", "stop_sequence"}:
        return "stop"
    return r


def model_slug(provider: str, model: str) -> str:
    base = f"{provider}_{model}"
    return base.replace(":", "_").replace("/", "_").replace(".", "-")


# ---------------------------------------------------------------------------
# Anbieter-Funktionen. Alle haben dieselbe Signatur und liefern dasselbe dict:
#   text, model_reported, done_reason_raw, prompt_eval_count, eval_count,
#   thinking_tokens, raw
#
# Die Aufrufe werden ausschliesslich aus params gebaut. Optionale Felder werden
# nur gesendet, wenn sie im Dict stehen -> params_json bleibt wahrheitsgemaess.
# ---------------------------------------------------------------------------

def call_anthropic(model: str, image_bytes: bytes, prompt: str, params: dict) -> dict:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    kwargs = {
        "model": model,
        "max_tokens": params["max_tokens"],
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }
    # Nur senden, was fuer dieses Modell auch vorgesehen ist.
    for optional in ("temperature", "thinking", "output_config"):
        if optional in params:
            kwargs[optional] = params[optional]

    resp = client.messages.create(**kwargs)

    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "output_tokens_details", None)

    return {
        "text": text,
        "model_reported": getattr(resp, "model", None),
        "done_reason_raw": getattr(resp, "stop_reason", None),
        "prompt_eval_count": getattr(usage, "input_tokens", None),
        "eval_count": getattr(usage, "output_tokens", None),
        "thinking_tokens": getattr(details, "thinking_tokens", None),
        "raw": to_jsonable(resp),
    }


def call_openai(model: str, image_bytes: bytes, prompt: str, params: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    kwargs = {
        "model": model,
        "max_output_tokens": params["max_output_tokens"],
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                {"type": "input_text", "text": prompt},
            ],
        }],
    }
    if "reasoning_effort" in params:
        kwargs["reasoning"] = {"effort": params["reasoning_effort"]}
    if "temperature" in params:
        kwargs["temperature"] = params["temperature"]

    resp = client.responses.create(**kwargs)

    # Bei Abbruch steht der Grund in incomplete_details, nicht in status.
    incomplete = getattr(resp, "incomplete_details", None)
    done_raw = getattr(incomplete, "reason", None) or getattr(resp, "status", None)

    usage = getattr(resp, "usage", None)
    details = getattr(usage, "output_tokens_details", None)

    return {
        "text": getattr(resp, "output_text", "") or "",
        "model_reported": getattr(resp, "model", None),
        "done_reason_raw": done_raw,
        "prompt_eval_count": getattr(usage, "input_tokens", None),
        "eval_count": getattr(usage, "output_tokens", None),
        "thinking_tokens": getattr(details, "reasoning_tokens", None),
        "raw": to_jsonable(resp),
    }


def call_google(model: str, image_bytes: bytes, prompt: str, params: dict) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config_kwargs = {"max_output_tokens": params["max_output_tokens"]}
    # temperature/top_p/top_k bewusst NICHT gesetzt: deprecated.

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(**config_kwargs),
    )

    candidates = getattr(resp, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    usage = getattr(resp, "usage_metadata", None)

    # resp.text kann None sein, wenn das Denken das Token-Budget aufgebraucht hat.
    text = getattr(resp, "text", None) or ""

    return {
        "text": text,
        "model_reported": getattr(resp, "model_version", None),
        "done_reason_raw": getattr(finish, "name", None) or finish,
        "prompt_eval_count": getattr(usage, "prompt_token_count", None),
        "eval_count": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "raw": to_jsonable(resp),
    }


DISPATCH = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "google": call_google,
}

REQUIRED_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
}


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

missing_params = [m["model"] for m in MODELS if m["model"] not in PARAMS]
if missing_params:
    raise SystemExit(f"Kein PARAMS-Eintrag fuer: {', '.join(missing_params)} - Abbruch.")

needed = {m["provider"] for m in MODELS}
missing_keys = [REQUIRED_KEYS[p] for p in needed if not os.environ.get(REQUIRED_KEYS[p])]
if missing_keys:
    raise SystemExit(f"Fehlende Keys in Umgebung/.env: {', '.join(missing_keys)} - Abbruch.")

all_images = sorted(IMAGES_DIR.glob("*.png"))
if not all_images:
    raise SystemExit(f"Keine PNGs gefunden in {IMAGES_DIR} - Abbruch.")

if IMAGE_FILTER:
    image_paths = [p for p in all_images if p.name == IMAGE_FILTER]
    if not image_paths:
        raise SystemExit(f"{IMAGE_FILTER} nicht in {IMAGES_DIR} - Abbruch.")
else:
    image_paths = all_images

print(f"Bilder:  {[p.name for p in image_paths]}")
print(f"Modelle: {[m['model'] for m in MODELS]}")
print(f"Laeufe je Kombination: {N_RUNS}\n")


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------

results = []
prompt_hash = sha256_text(PROMPT_GEN)

for entry in MODELS:
    provider = entry["provider"]
    model = entry["model"]
    params = PARAMS[model]          # <- nach Modell, nicht nach Anbieter
    slug = model_slug(provider, model)

    print(f"\n===== {provider} / {model} (Prompt {PROMPT_ID}) =====")

    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        image_hash = sha256_bytes(image_bytes)

        for run_index in range(N_RUNS):
            print(f"--- {model} auf {image_path.name} "
                  f"(Lauf {run_index + 1}/{N_RUNS}) ---")

            suffix = "" if N_RUNS == 1 else f"__run{run_index}"
            record = {
                "provider": provider,
                "model": model,
                "role": entry["role"],
                "image": image_path.name,
                "run_index": run_index,
                "prompt_id": PROMPT_ID,
                "prompt_sha256": prompt_hash,
                "image_sha256": image_hash,
                "params_json": json.dumps(params, sort_keys=True),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            start = time.time()
            last_error = None
            out = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    out = DISPATCH[provider](model, image_bytes, PROMPT_GEN, params)
                    break
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    # 400er sind Konfigurationsfehler, kein Wiederholen noetig.
                    if "400" in msg or "invalid_request" in msg:
                        print(f"  Konfigurationsfehler, kein Retry: {msg[:300]}")
                        break
                    if attempt == MAX_RETRIES:
                        break
                    wait = RETRY_BASE_SLEEP * attempt
                    print(f"  Versuch {attempt} fehlgeschlagen ({type(e).__name__}), "
                          f"warte {wait}s ...")
                    time.sleep(wait)

            elapsed = time.time() - start

            if out is None:
                record.update({
                    "model_reported": None,
                    "done_reason": None,
                    "done_reason_raw": None,
                    "prompt_eval_count": None,
                    "eval_count": None,
                    "thinking_tokens": None,
                    "seconds": round(elapsed, 1),
                    "output_path": None,
                    "raw_path": None,
                    "error": f"{type(last_error).__name__}: {last_error}",
                })
                results.append(record)
                print(f"FEHLER: {last_error}\n")
                continue

            out_path = TRANSCRIPTIONS_DIR / f"{image_path.stem}__{slug}{suffix}.txt"
            out_path.write_text(out["text"], encoding="utf-8")

            raw_path = RAW_DIR / f"{image_path.stem}__{slug}{suffix}.json"
            raw_path.write_text(
                json.dumps(out["raw"], ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            record.update({
                "model_reported": out["model_reported"],
                "done_reason": normalize_done_reason(out["done_reason_raw"]),
                "done_reason_raw": out["done_reason_raw"],
                "prompt_eval_count": out["prompt_eval_count"],
                "eval_count": out["eval_count"],
                "thinking_tokens": out["thinking_tokens"],
                "seconds": round(elapsed, 1),
                "output_path": str(out_path),
                "raw_path": str(raw_path),
                "error": None,
            })
            results.append(record)

            print(out["text"])
            print(f"Dauer: {elapsed:.1f}s | done_reason: {record['done_reason']} "
                  f"| thinking_tokens: {out['thinking_tokens']} "
                  f"| gemeldetes Modell: {out['model_reported']}\n")


# ---------------------------------------------------------------------------
# Ergebnisse
# ---------------------------------------------------------------------------

df = pd.DataFrame(results)
ergebnis_csv = TABELLEN_DIR / "frontier_results.csv"
df.to_csv(ergebnis_csv, index=False)

print(df[["provider", "model", "image", "run_index", "done_reason",
          "eval_count", "thinking_tokens", "seconds"]].to_string(index=False))
print(f"\nGeschrieben: {ergebnis_csv}")

# Angefordertes vs. tatsaechlich gerechnetes Modell - Aliase verschleiern das sonst.
mismatch = df[df["model_reported"].notna() & (df["model_reported"] != df["model"])]
if not mismatch.empty:
    print("\n! HINWEIS: gemeldetes Modell weicht von der angeforderten ID ab.")
    print("  Fuer den Methodenteil die Spalte model_reported verwenden.")
    print(mismatch[["model", "model_reported"]].drop_duplicates().to_string(index=False))

# Thinking sollte bei den Anthropic-Modellen 0 sein - Gegenprobe zum Vorlauf.
thinking_on = df[df["thinking_tokens"].fillna(0) > 0]
if not thinking_on.empty:
    print("\n! HINWEIS: diese Laeufe haben gedacht (thinking_tokens > 0).")
    print("  Bei Gemini erwartet, bei Anthropic/OpenAI nicht - dann Setting pruefen.")
    print(thinking_on[["model", "thinking_tokens"]].to_string(index=False))

# Abgeschnittene Laeufe sind fuer die CER-Auswertung unbrauchbar.
truncated = df[df["done_reason"].eq("length")]
if not truncated.empty:
    print("\n! ABGESCHNITTEN (done_reason='length') - diese Zeilen nicht auswerten:")
    print(truncated[["model", "image", "run_index", "eval_count"]].to_string(index=False))

errors = df[df["error"].notna()]
if not errors.empty:
    print("\n! FEHLGESCHLAGEN:")
    print(errors[["provider", "model", "image", "error"]].to_string(index=False))