import ollama
import time
import pandas as pd
from pathlib import Path

# Kleinste Modelle zuerst, das große deepseek-ocr zuletzt (rein zur Sicherheit,
# die Reihenfolge ist dank Entladen zwischen den Modellen aber egal).
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
PROMPT = "Transkribiere den gesamten sichtbaren Text auf diesem Bild, exakt und ohne Kommentar oder Zusätze."

# Alle Pfade relativ zur Projektwurzel (eine Ebene über src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRANSCRIPTIONS_DIR = OUTPUT_DIR / "transcriptions"
TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(IMAGES_DIR.glob("*.png"))
if not image_paths:
    print(f"Keine PNGs gefunden in {IMAGES_DIR}")


# ---------------------------------------------------------------------------
# RAM-Verwaltung: Modelle gezielt aus dem Speicher entladen
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """':latest' abschneiden, damit Tag-Varianten zusammenpassen."""
    return name[:-7] if name.endswith(":latest") else name


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
# Hauptschleife
# ---------------------------------------------------------------------------

results = []

for model in MODELS:
    print(f"\n===== Modell: {model} =====")
    for image_path in image_paths:
        print(f"--- Teste {model} auf {image_path.name} ---")
        start = time.time()
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [str(image_path)],
                }],
            
                options={
                    "temperature": 0,   # greedy -> deterministische Transkription
                    "seed": 42,         # nur bei temperature>0 wirksam, aber schadet nicht
                    "top_k": 1,                 # bei greedy wirkungslos, nur zur Dokumentation
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,      # Ollama-Default 1.1 -> bestraft legitime Wiederholungen
                    "repeat_last_n": 0,
                    "presence_penalty": 0.0,    # qwen3.5 bringt 1.5 als Modell-Default mit
                    "frequency_penalty": 0.0,
                    "num_predict": 4096,

                },
                think=False,
            )
            

            text = response["message"]["content"]
            elapsed = time.time() - start

            # Transkription je Bild in output/ ablegen, z.B. output/anzeige01__qwen3-vl_2b.txt
            model_slug = model.replace(":", "_").replace("/", "_")
            out_path = TRANSCRIPTIONS_DIR / f"{image_path.stem}__{model_slug}.txt"
            out_path.write_text(text, encoding="utf-8")

            results.append({
                "model": model,
                "image": image_path.name,
                "seconds": round(elapsed, 1),
                "output_path": str(out_path),
            })
            print(text)
            print(f"Dauer: {elapsed:.1f}s\n")
        except Exception as e:
            results.append({
                "model": model,
                "image": image_path.name,
                "seconds": None,
                "output_path": None,
                "error": str(e),
            })
            print(f"FEHLER bei {image_path.name}: {e}\n")

    # Erst wenn ALLE Bilder dieses Modells durch sind: RAM freigeben, bevor das
    # nächste Modell geladen wird -> nie zwei Modelle gleichzeitig im Speicher.
    unload_model(model)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / "ocr_results.csv", index=False)
print(df[["model", "image", "seconds"]])