"""
Berechnet CER und WER für alle OCR-Transkriptionen in output/transcriptions/
gegen die Goldstandard-Transkriptionen in data/gold/.

Zwei Auswertungsstufen (beide werden berechnet und in eine CSV geschrieben):

  "roh"  – Modellausgabe wie geliefert (nur Unicode-NFC).
           Misst Transkriptionsgüte UND Instruction-Following gemeinsam.
           Hier schlagen Markdown-Rahmen, Trennlinien und Wiederholungen
           voll durch (= Fehlertyp E, generative Artefakte).

  "norm" – Layout- und Markdown-Artefakte deterministisch entfernt,
           identisch angewandt auf Gold UND Modellausgabe.
           Misst die Transkriptionsgüte allein.

Die Differenz zwischen beiden Stufen ist selbst ein Ergebnis: sie zeigt,
wie viel Nachbearbeitung ein Modell braucht, um brauchbar zu sein.

WICHTIG – was die Normalisierung bewusst NICHT anfasst:
  - langes ſ, Ligaturen, k/t, n/u  -> genau das ist der Messgegenstand (Typ A)
  - historische Orthografie (z.B. Thür/Tür, Cigarren/Zigarren)  -> Typ C
  - Groß-/Kleinschreibung (LOWERCASE = False)  -> Teil der Transkription
Diese Artefakte dürfen nicht wegnormalisiert werden, sonst misst man nichts mehr.
"""

import re
import subprocess
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd
from dinglehopper import character_error_rate, word_error_rate

# --- Pfade (Projektwurzel = eine Ebene über src/, wie in ollama_ocr.py) --------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
TRANSCRIPTIONS_DIR = PROJECT_ROOT / "output" / "transcriptions"
METRICS_DIR = PROJECT_ROOT / "output" / "metrics"
NORMALIZED_DIR = METRICS_DIR / "normalized"      # normalisierte Texte zum Nachprüfen
DINGLEHOPPER_DIR = METRICS_DIR / "dinglehopper"  # HTML-Diffs für die Fehlertypologie

for d in (METRICS_DIR, NORMALIZED_DIR, DINGLEHOPPER_DIR):
    d.mkdir(parents=True, exist_ok=True)

ENCODING = "utf-8"
LOWERCASE = False          # Fraktur: Groß-/Kleinschreibung gehört zur Transkription
DEDUPLICATE = True         # doppelt ausgegebene Blöcke (z.B. GLM) zusammenfassen
SAVE_HTML_DIFF_REPORTS = True


# --- Normalisierung -----------------------------------------------------------

SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")                                  # <|im_end|>
MARKDOWN_FENCE = re.compile(r"^\s*(```|~~~).*$", re.MULTILINE)              # ```text
HORIZONTAL_RULE = re.compile(r"^\s*([-*_])\s*(?:\1\s*){2,}$", re.MULTILINE) # --- *** ___
HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)                    # ## Überschrift
BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)                      # > Zitat
LIST_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)                 # - Punkt
EMPHASIS = re.compile(r"(\*\*|\*|__)")                                      # **fett**
TABLE_PIPE = re.compile(r"[|]")


def normalize(text: str) -> str:
    """Entfernt Markdown- und Layout-Artefakte. Wird auf Gold UND Hypothese angewandt."""
    text = unicodedata.normalize("NFC", text)
    text = SPECIAL_TOKEN.sub("", text)   # muss VOR TABLE_PIPE laufen
    text = MARKDOWN_FENCE.sub("", text)
    text = HORIZONTAL_RULE.sub("", text)
    text = HEADING.sub("", text)
    text = BLOCKQUOTE.sub("", text)
    text = LIST_BULLET.sub("", text)
    text = EMPHASIS.sub("", text)
    text = TABLE_PIPE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)   # alle Umbrüche/Leerzeilen -> ein Leerzeichen
    text = text.strip()
    if LOWERCASE:
        text = text.lower()
    return text


def deduplicate(text: str) -> tuple[str, int]:
    """Wiederholt ein Modell denselben Block mehrfach, wird nur der erste behalten.
    Rückgabe: (bereinigter Text, Anzahl Blockwiederholungen; 1 = keine)."""
    anchor = text[:60]
    n = text.count(anchor)
    if n <= 1:
        return text, 1
    return text[:text.find(anchor, 1)].strip(), n


# --- Hilfsfunktionen ----------------------------------------------------------

def parse_filename(txt_path: Path) -> tuple[str, str]:
    """'{image_stem}__{model_slug}.txt' -> (image_stem, model_slug)."""
    stem = txt_path.stem
    if "__" not in stem:
        raise ValueError(f"Kann Dateiname nicht in Bild/Modell zerlegen: {txt_path.name}")
    image_stem, model_slug = stem.split("__", 1)
    return image_stem, model_slug


def metrics(gold: str, hyp: str) -> dict:
    # Zeichen-Operationen aufschluesseln: CER allein verdeckt, ob ein Modell
    # falsch liest (S), Material verliert (D) oder dazuerfindet (I).
    o = jiwer.process_characters(gold, hyp)
    return {
        "cer_jiwer": round(jiwer.cer(gold, hyp), 4),
        "wer_jiwer": round(jiwer.wer(gold, hyp), 4),
        "cer_dinglehopper": round(character_error_rate(gold, hyp), 4),
        "wer_dinglehopper": round(word_error_rate(gold, hyp), 4),
        "S": o.substitutions,
        "D": o.deletions,
        "I": o.insertions,
        "N": o.substitutions + o.deletions + o.hits,   # Referenzlaenge
    }


def save_html_diff_report(gold_path: Path, hyp_path: Path, report_prefix: str) -> None:
    """dinglehopper-CLI nur für den visuellen HTML-Diff (Basis der Fehlertypologie)."""
    try:
        subprocess.run(
            ["dinglehopper", str(gold_path), str(hyp_path), report_prefix,
             str(DINGLEHOPPER_DIR), "--plain-encoding", ENCODING],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ! dinglehopper-Report fehlgeschlagen für {report_prefix}: {e.stderr}")


# --- Hauptlauf ----------------------------------------------------------------

def main() -> None:
    txt_paths = sorted(TRANSCRIPTIONS_DIR.glob("*.txt"))
    if not txt_paths:
        print(f"Keine Transkriptionen gefunden in {TRANSCRIPTIONS_DIR}")
        return

    rows = []
    for txt_path in txt_paths:
        image_stem, model_slug = parse_filename(txt_path)
        gold_path = GOLD_DIR / f"{image_stem}.txt"
        if not gold_path.exists():
            print(f"  ! Kein Gold-Transkript für '{image_stem}' -> {txt_path.name} übersprungen")
            continue

        gold_raw = unicodedata.normalize("NFC", gold_path.read_text(encoding=ENCODING))
        hyp_raw = unicodedata.normalize("NFC", txt_path.read_text(encoding=ENCODING))

        gold_norm = normalize(gold_raw)
        hyp_norm = normalize(hyp_raw)
        wiederholungen = 1
        if DEDUPLICATE:
            hyp_norm, wiederholungen = deduplicate(hyp_norm)

        # Normalisierte Texte sichern -> nachvollziehbar, was tatsächlich gemessen wurde
        norm_hyp_path = NORMALIZED_DIR / f"{txt_path.stem}.txt"
        norm_hyp_path.write_text(hyp_norm, encoding=ENCODING)
        gold_norm_path = NORMALIZED_DIR / f"GOLD__{image_stem}.txt"
        gold_norm_path.write_text(gold_norm, encoding=ENCODING)

        for level, (g, h) in {"roh": (gold_raw, hyp_raw),
                              "norm": (gold_norm, hyp_norm)}.items():
            rows.append({
                "model": model_slug,
                "image": image_stem,
                "stufe": level,
                **metrics(g, h),
                "wiederholungen": wiederholungen,            })

        if SAVE_HTML_DIFF_REPORTS:
            save_html_diff_report(gold_norm_path, norm_hyp_path, txt_path.stem)

        flag = f"  [{wiederholungen}x wiederholt]" if wiederholungen > 1 else ""
        roh, norm = rows[-2], rows[-1]
        print(f"{model_slug:30s} {image_stem:12s} "
              f"roh CER {roh['cer_dinglehopper']:.4f} WER {roh['wer_dinglehopper']:.4f}  ->  "
              f"norm CER {norm['cer_dinglehopper']:.4f} WER {norm['wer_dinglehopper']:.4f}{flag}")

    if not rows:
        print("Keine Zeile berechnet (fehlende Gold-Dateien?).")
        return

    df = pd.DataFrame(rows)
    out_csv = METRICS_DIR / "cer_wer_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} Zeilen gespeichert in {out_csv}")

    print("\n--- Vergleich (dinglehopper), sortiert nach CER norm ---")
    pivot = df.pivot_table(index="model", columns="stufe",
                           values=["cer_dinglehopper", "wer_dinglehopper"])
    print(pivot.sort_values(("cer_dinglehopper", "norm")).round(4))


if __name__ == "__main__":
    main()