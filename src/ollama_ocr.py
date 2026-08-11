import ollama
import json
import time
import pandas as pd
from pathlib import Path

# Kleinste Modelle zuerst, das große deepseek-ocr zuletzt (rein zur Sicherheit,
# die Reihenfolge ist dank Entladen zwischen den Modellen aber egal).
MODELS = [
    "qwen3-vl:4b-instruct",
    "qwen3-vl:8b-instruct",
    "gemma4:e4b",
    "qwen3.5:4b",
    "maternion/lightonocr-2:1b",
    "glm-ocr:bf16",
    "deepseek-ocr:3b",
    "minicpm-v4.5:8b",
]

# ---------------------------------------------------------------------------
# Prompts: einheitlich innerhalb der Generalisten, Task-Praefixe laut Modelcard
# bei den OCR-Spezialisten. Herkunft je Prompt in PROMPT_META dokumentiert.
# ---------------------------------------------------------------------------

PROMPT_GEN = (
    "Transkribiere den gesamten sichtbaren Text auf diesem Bild, exakt und ohne Kommentar, ohne Formatierung oder Zusätze."
)

PROMPTS = {
    "qwen3-vl:4b-instruct":               ("P-GEN-01", PROMPT_GEN),
    "qwen3-vl:8b-instruct":               ("P-GEN-01", PROMPT_GEN),
    "gemma4:e4b":                ("P-GEN-01", PROMPT_GEN),
    "qwen3.5:4b":                ("P-GEN-01", PROMPT_GEN),
    "minicpm-v4.5:8b":           ("P-GEN-01", PROMPT_GEN),
    "maternion/lightonocr-2:1b": ("P-OCR-LIGHTON-01", ""),
    "glm-ocr:bf16":              ("P-OCR-GLM-01", "Text Recognition:"),
    "deepseek-ocr:3b":           ("P-OCR-DSK-01", "Free OCR."),
}

PROMPT_META = {
    "P-GEN-01":         "eigene Formulierung; einheitlich fuer alle generalistischen VLMs",
    "P-OCR-LIGHTON-01": "Modelcard lightonai/LightOnOCR-2-1B: Beispiel uebergibt nur das Bild, keinen Text",
    "P-OCR-GLM-01":     "Modelcard zai-org/GLM-OCR: 'Text Recognition:'",
    "P-OCR-DSK-01":     "Modelcard deepseek-ai/DeepSeek-OCR: '<image>\\nFree OCR.'; Bildtoken setzt Ollama selbst",
}

# Modelle mit thinking-Capability laut Modellinventar (Tabelle A).
# Nur diese bekommen think=False uebergeben.
THINKING_MODELS = {
    "qwen3-vl:4b-instruct",
    "qwen3-vl:8b-instruct",
    "gemma4:e4b",
    "qwen3.5:4b",
    "minicpm-v4.5:8b",
}

GEN_OPTIONS = {
    "temperature": 0,        # greedy decoding -> traegt den Determinismus
    "seed": 42,              # bei temperature=0 wirkungslos, nur zur Dokumentation
    "top_k": 1,              # bei greedy wirkungslos, nur zur Dokumentation
    "top_p": 1.0,            # dito
    "repeat_penalty": 1.0,   # Ollama-Default 1.1 bestraft legitime Wiederholungen
    "repeat_last_n": 0,
    "presence_penalty": 0.0, # qwen3.5 braechte sonst als einziges Modell 1.5 mit
    "frequency_penalty": 0.0,
    "num_predict": 1024,     # verhindert Abschneiden langer Transkriptionen
    "num_ctx": 8192,         # groesster Wert, den alle acht Modelle unterstuetzen
}

# Alle Pfade relativ zur Projektwurzel (eine Ebene über src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "transcriptions"
RAW_DIR = OUTPUT_DIR / "raw"
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(IMAGES_DIR.glob("*.png"))
if not image_paths:
    raise SystemExit(f"Keine PNGs gefunden in {IMAGES_DIR} - Abbruch.")


# ---------------------------------------------------------------------------
# RAM-Verwaltung: Modelle gezielt aus dem Speicher entladen
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """':bf16' abschneiden, damit Tag-Varianten zusammenpassen."""
    return name[:-7] if name.endswith(":bf16") else name


def running_models():
    """Namen der aktuell im RAM geladenen Modelle (robust gegen ollama-Versionen)."""
    try:
        resp = ollama.ps()
    except Exception:
        return []
    models = getattr(resp, "models", None)
    if models is None and isinstance(resp, dict):
        models = resp.get("models", [])
    names = []
    for m in (models or []):
        name = getattr(m, "model", None) or getattr(m, "name", None)
        if name is None and isinstance(m, dict):
            name = m.get("model") or m.get("name")
        if name:
            names.append(name)
    return names


def is_loaded(model: str) -> bool:
    target = _norm(model)
    return any(_norm(n) == target for n in running_models())


def unload_model(model: str, timeout: int = 120):
    """Entlädt ein Modell sofort aus dem RAM und wartet, bis es wirklich weg ist."""
    if not is_loaded(model):
        return
    print(f"  -> Entlade {model} aus dem RAM ...")
    try:
        # keep_alive=0 sagt Ollama: Modell direkt nach dieser (leeren) Anfrage entladen
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception as e:
        print(f"  (Hinweis beim Entladen von {model}: {e})")
    # Warten, bis der Speicher tatsächlich frei ist – erst dann das nächste Modell laden
    start = time.time()
    while time.time() - start < timeout:
        if not is_loaded(model):
            print(f"  -> {model} ist entladen.\n")
            return
        time.sleep(1)
    print(f"  ! Warnung: {model} ist nach {timeout}s noch geladen.\n")


# ---------------------------------------------------------------------------
# Antwort-Auslesen: die ollama-Bibliothek liefert je nach Version ein Objekt
# oder ein dict zurueck. Beide Zugriffswege abdecken.
# ---------------------------------------------------------------------------

def resp_field(resp, key, default=None):
    try:
        return resp[key]
    except Exception:
        return getattr(resp, key, default)


def resp_to_dict(resp):
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return dict(resp)


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------

results = []

for model in MODELS:
    prompt_id, prompt_text = PROMPTS[model]
    print(f"\n===== Modell: {model} (Prompt {prompt_id}) =====")
    for image_path in image_paths:
        print(f"--- Teste {model} auf {image_path.name} ---")
        model_slug = model.replace(":", "_").replace("/", "_")
        start = time.time()
        try:
            chat_kwargs = {}
            if model in THINKING_MODELS:
                chat_kwargs["think"] = False

            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt_text,
                    "images": [str(image_path)],
                }],
                options=GEN_OPTIONS,
                **chat_kwargs,
            )

            text = response["message"]["content"]
            elapsed = time.time() - start

            # Transkription je Bild in output/ ablegen, z.B. output/anzeige01__qwen3-vl_4b.txt
            out_path = TRANSCRIPTIONS_DIR / f"{image_path.stem}__{model_slug}.txt"
            out_path.write_text(text, encoding="utf-8")

            # Vollstaendige Roh-Antwort als Belegkette archivieren
            raw_path = RAW_DIR / f"{image_path.stem}__{model_slug}.json"
            raw_path.write_text(
                json.dumps(resp_to_dict(response), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            results.append({
                "model": model,
                "model_reported": resp_field(response, "model"),
                "image": image_path.name,
                "prompt_id": prompt_id,
                "done_reason": resp_field(response, "done_reason"),
                "prompt_eval_count": resp_field(response, "prompt_eval_count"),
                "eval_count": resp_field(response, "eval_count"),
                "seconds": round(elapsed, 1),
                "output_path": str(out_path),
                "raw_path": str(raw_path),
            })
            print(text)
            print(f"Dauer: {elapsed:.1f}s\n")
        except Exception as e:
            results.append({
                "model": model,
                "model_reported": None,
                "image": image_path.name,
                "prompt_id": prompt_id,
                "done_reason": None,
                "prompt_eval_count": None,
                "eval_count": None,
                "seconds": None,
                "output_path": None,
                "raw_path": None,
                "error": str(e),
            })
            print(f"FEHLER bei {image_path.name}: {e}\n")

    # Erst wenn ALLE Bilder dieses Modells durch sind: RAM freigeben, bevor das
    # nächste Modell geladen wird -> nie zwei Modelle gleichzeitig im Speicher.
    unload_model(model)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "ocr_results.csv", index=False)
print(df[["model", "image", "prompt_id", "done_reason", "eval_count", "seconds"]])

# Abgeschnittene Laeufe sind fuer die CER-Auswertung unbrauchbar -> sofort sichtbar machen
truncated = df[df.get("done_reason").eq("length")] if "done_reason" in df else None
if truncated is not None and not truncated.empty:
    print("\n! ABGESCHNITTEN (done_reason='length') - diese Zeilen nicht auswerten:")
    print(truncated[["model", "image", "eval_count"]])