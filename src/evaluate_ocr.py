"""
Berechnet CER und WER für alle OCR-Transkriptionen in output/transcriptions/
gegen die Goldstandard-Transkriptionen in data/gold/.

Zwei Auswertungsstufen (beide werden berechnet und in eine CSV geschrieben):

  "roh"  – Modellausgabe wie geliefert (nur Unicode-NFC, ohne Rand-Whitespace).
           Misst Transkriptionsgüte UND Instruction-Following gemeinsam.
           Hier schlagen HTML-Gerüst, Markdown-Rahmen, Bildplatzhalter und
           Wiederholungen voll durch (= Fehlertyp E, generative Artefakte).

  "norm" – Auszeichnung und Layout-Gerüst deterministisch entfernt,
           identisch angewandt auf Gold UND Modellausgabe.
           Misst die Transkriptionsgüte allein.

Die Differenz zwischen beiden Stufen ist selbst ein Ergebnis: sie zeigt,
wie viel Nachbearbeitung ein Modell braucht, um brauchbar zu sein. Die
Spalte 'artefakt_anteil' beziffert sie direkt (Anteil der Zeichen einer
Modellausgabe, die reines Gerüst waren).

LEITREGEL DER NORMALISIERUNG
  Entfernt wird nur, was keinen Anspruch darüber erhebt, WELCHE ZEICHEN
  auf der Vorlage stehen: Tags, Markdown-Syntax, Bildplatzhalter,
  Zierglyphen, unsichtbare Steuerzeichen. Alles, was eine Lesart des
  Drucks behauptet, bleibt stehen.

  Ein Bildplatzhalter wird ersatzlos gelöscht, nicht durch Text ersetzt:
  hat das Modell einen Textblock (z.B. die Logo-Inschrift in anzeige01)
  durch ![image](...) ersetzt, soll das als Löschung zählen – die
  Auslassung ist ein echter Transkriptionsfehler, das Platzhalter-Token
  selbst aber keine Einfügung.

WICHTIG – was die Normalisierung bewusst NICHT anfasst:
  - langes ſ, Ligaturen, k/t, n/u  -> genau das ist der Messgegenstand (Typ A)
  - historische Orthografie (z.B. Thür/Tür, Cigarren/Zigarren)  -> Typ C
  - Groß-/Kleinschreibung (LOWERCASE = False)  -> Teil der Transkription
  - Trennstriche am Zeilenende, Sperrsatz, Interpunktion  -> Typ B
  - ° (U+00B0), Gedankenstriche, Anführungszeichen  -> echte Drucktypen
Diese Artefakte dürfen nicht wegnormalisiert werden, sonst misst man nichts mehr.

Am Ende läuft eine Artefakt-Kontrolle: jedes Zeichen, das nach der
Normalisierung in einer Modellausgabe steht, aber in keinem Gold vorkommt,
wird gemeldet. So fällt neues Gerüst eines neuen Modells auf, statt still
in die CER zu wandern.
"""

import html
import re
import subprocess
import sys
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
# Sammelordner für alle Tabellen-Ausgaben des Projekts (CSV/MD/JSON).
TABELLEN_DIR = PROJECT_ROOT / "output" / "tabellen"

for d in (METRICS_DIR, NORMALIZED_DIR, DINGLEHOPPER_DIR, TABELLEN_DIR):
    d.mkdir(parents=True, exist_ok=True)

ENCODING = "utf-8"
LOWERCASE = False            # Fraktur: Groß-/Kleinschreibung gehört zur Transkription
ENTFERNE_ZIERGLYPHEN = True  # ▽ ✦ ❧ usw.: Vignetten, die das Gold nicht transkribiert
ENTFERNE_NUMMERIERTE_LISTEN = False  # Begründung bei LIST_NUMMER weiter unten
DEDUPLICATE = True           # doppelt ausgegebene Blöcke zusammenfassen
SAVE_HTML_DIFF_REPORTS = True
MIN_BLOCK_TOKEN = 8          # kürzere "Wiederholungen" sind meist echter Satzinhalt


# --- Normalisierung -----------------------------------------------------------
# Reihenfolge der Anwendung ist in normalize() festgelegt und dort kommentiert.

# 1. Steuer- und Sondertokens
SPECIAL_TOKEN = re.compile(r"<\|[^|]*\|>")                       # <|im_end|>
UNSICHTBAR = re.compile(          # weiches Trennzeichen, ZWSP/ZWJ,
    "[\u00ad\u200b-\u200f\u2028\u2029\u2060\ufeff]")  # Richtungsmarken, BOM

# 2. HTML-Konstrukte, deren Inhalt nie Transkription ist
HTML_KOMMENTAR = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_MIT_INHALT = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>",
                             re.DOTALL | re.IGNORECASE)

# 3. Markdown-Blockstruktur (braucht die originale Zeilenstruktur)
MD_FENCE = re.compile(r"^[ \t]*(?:```|~~~).*$", re.MULTILINE)          # ```text
MD_TABELLENTRENNER = re.compile(                                       # |---|:--:|
    r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$",
    re.MULTILINE)
HORIZONTAL_RULE = re.compile(r"^[ \t]*([-*_=])[ \t]*(?:\1[ \t]*){2,}$",
                             re.MULTILINE)                             # --- *** ___ ===
HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE)         # ## Überschrift
BLOCKQUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)           # > Zitat
LIST_BULLET = re.compile(r"^[ \t]{0,3}[-*+][ \t]+", re.MULTILINE)      # - Punkt
LIST_NUMMER = re.compile(r"^[ \t]{0,3}\d{1,2}[.)][ \t]+", re.MULTILINE)  # 1. Punkt
# Spiegelstriche (LIST_BULLET) müssen weg: LightOnOCR-2 setzt die Spirituosen-
# liste in anzeige01 als "- Kümmel". Keine Golddatei beginnt eine Zeile mit "- ".
# LIST_NUMMER ist dagegen per Default AUS (ENTFERNE_NUMMERIERTE_LISTEN): kein
# Modell hat je eine nummerierte Liste erzeugt, dafür stehen in Anzeigen reihen-
# weise Haus- und Straßennummern. Fiele "1." am Zeilenanfang weg, verschwände
# eine Verwechslung "1. Etage"/"7. Etage" spurlos aus der Messung.

# 4. Markdown-Inline
MD_BILD = re.compile(r"!\[[^\]]*\]\([^)]*\)")                # ![alt](datei) -> weg
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")               # [Text](url)   -> Text
MD_REF_LINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")          # [Text][ref]   -> Text
EMPHASIS = re.compile(r"\*\*|\*|__|~~|`")                    # **fett** `code`

# 5. HTML-Tags. Block-Tags trennen Textblöcke und werden zu einem Leerzeichen,
#    damit "Landwirtschafts-<br>Zeitschrift" nicht zu einem Wort verschmilzt.
#    Inline-Tags stehen im Wortinneren und verschwinden ersatzlos.
HTML_BLOCK_TAG = re.compile(
    r"</?(?:div|p|br|hr|h[1-6]|li|ul|ol|dl|dt|dd|table|thead|tbody|tfoot|tr|td|th"
    r"|caption|colgroup|col|section|article|header|footer|main|nav|aside|center"
    r"|blockquote|pre|figure|figcaption|form|img|body|html|head)\b[^>]*/?>",
    re.IGNORECASE)
HTML_TAG = re.compile(r"</?[a-zA-Z!?][^>]*>")

# 6. Tabellenrahmen
TABLE_PIPE = re.compile(r"\|")

# 7. Zierglyphen: Vignetten, Trennornamente, Pfeile, Kästchen. Bewusst über
#    Unicode-Blöcke definiert und NICHT über die Kategorie So – sonst fiele
#    ° (U+00B0) aus "Aquavit zu 40, 50 und 60°" mit weg. Gedankenstriche
#    (U+2013/2014) und Anführungszeichen liegen ebenfalls außerhalb.
ZIERGLYPHE = re.compile(
    "["
    "\u2190-\u21ff"   # Pfeile
    "\u2300-\u23ff"   # technische Zeichen
    "\u2500-\u257f"   # Rahmenzeichnung
    "\u2580-\u259f"   # Blockelemente
    "\u25a0-\u25ff"   # geometrische Formen  (U+25BD "weisses Dreieck" in anzeige01)
    "\u2600-\u27bf"   # Symbole und Dingbats
    "\u2b00-\u2bff"   # weitere Formen und Pfeile
    "\U0001f300-\U0001faff"   # Emoji
    "]")


def normalize(text: str) -> str:
    """Entfernt Auszeichnung und Layout-Gerüst. Wird auf Gold UND Hypothese
    angewandt. Siehe Leitregel im Modul-Docstring."""
    text = unicodedata.normalize("NFC", text)
    text = UNSICHTBAR.sub("", text)
    text = SPECIAL_TOKEN.sub("", text)        # muss VOR HTML_TAG/TABLE_PIPE laufen

    text = HTML_KOMMENTAR.sub(" ", text)
    text = HTML_MIT_INHALT.sub(" ", text)

    # Blockstruktur zuerst, solange die Zeilenanfänge noch intakt sind
    text = MD_FENCE.sub("", text)
    text = MD_TABELLENTRENNER.sub("", text)   # vor TABLE_PIPE und HORIZONTAL_RULE
    text = HORIZONTAL_RULE.sub("", text)
    text = HEADING.sub("", text)
    text = BLOCKQUOTE.sub("", text)
    text = LIST_BULLET.sub("", text)
    if ENTFERNE_NUMMERIERTE_LISTEN:
        text = LIST_NUMMER.sub("", text)

    text = MD_BILD.sub("", text)              # vor MD_LINK: sonst bleibt "!alt"
    text = MD_LINK.sub(r"\1", text)
    text = MD_REF_LINK.sub(r"\1", text)
    text = EMPHASIS.sub("", text)

    text = HTML_BLOCK_TAG.sub(" ", text)
    text = HTML_TAG.sub("", text)
    text = html.unescape(text)                # nach dem Tag-Strippen, sonst
                                              # wird aus &lt;div&gt; ein Tag
    text = TABLE_PIPE.sub(" ", text)
    if ENTFERNE_ZIERGLYPHEN:
        text = ZIERGLYPHE.sub("", text)

    text = re.sub(r"\s+", " ", text)   # alle Umbrüche/Leerzeilen -> ein Leerzeichen
    text = text.strip()
    if LOWERCASE:
        text = text.lower()
    return text


def deduplicate(text: str) -> tuple[str, int]:
    """Kollabiert eine echte Blockwiederholung auf ihr erstes Vorkommen.

    Gesucht wird die kürzeste Periode p (in Token), für die der gesamte Text
    aus Wiederholungen der ersten p Token besteht; die letzte Wiederholung
    darf abgeschnitten sein. Verlangt wird exakte Übereinstimmung – eine
    entgleiste Generierung wiederholt sich wörtlich.

    Die frühere Variante verglich stattdessen die ersten 60 ZEICHEN mit dem
    Rest des Textes. Bei HTML-Ausgaben beginnen mehrere Blöcke identisch
    ('<div style="text-align: center; font-size: 24pt; font-weight: bo'),
    und der Text wurde mitten im Dokument abgeschnitten: anzeige04 verlor so
    seinen kompletten Fließtext und bekam 371 Löschungen zugeschrieben, die
    das Modell nie produziert hat.

    Rückgabe: (bereinigter Text, Anzahl Blockwiederholungen; 1 = keine)."""
    token = text.split()
    n = len(token)
    if n < 2 * MIN_BLOCK_TOKEN:
        return text, 1
    for p in range(MIN_BLOCK_TOKEN, n // 2 + 1):
        if all(token[i] == token[i - p] for i in range(p, n)):
            return " ".join(token[:p]), -(-n // p)   # aufgerundet
    return text, 1


# --- Hilfsfunktionen ----------------------------------------------------------

def kompakt(text: str) -> str:
    """Rohtext nur auf einheitlichen Whitespace gebracht – Bezugsgröße, um zu
    beziffern, wie viele Zeichen die Normalisierung tatsächlich entfernt hat."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


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


def dinglehopper_cli() -> str:
    """Pfad zur CLI. Erst neben dem laufenden Interpreter suchen – so klappt der
    Aufruf auch ohne aktiviertes venv (./env/bin/python src/evaluate_ocr.py)."""
    neben_python = Path(sys.executable).with_name("dinglehopper")
    return str(neben_python) if neben_python.exists() else "dinglehopper"


def save_html_diff_report(gold_path: Path, hyp_path: Path, report_prefix: str) -> None:
    """dinglehopper-CLI nur für den visuellen HTML-Diff (Basis der Fehlertypologie)."""
    try:
        subprocess.run(
            [dinglehopper_cli(), str(gold_path), str(hyp_path), report_prefix,
             str(DINGLEHOPPER_DIR), "--plain-encoding", ENCODING],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ! dinglehopper-Report fehlgeschlagen für {report_prefix}: {e.stderr}")
    except FileNotFoundError:
        # nur der optionale HTML-Diff fehlt -> Auswertung darf trotzdem durchlaufen
        print("  ! dinglehopper-CLI nicht gefunden -> HTML-Diffs übersprungen")


def artefakt_kontrolle(gold_texte: dict[str, str], hyp_texte: dict[str, str]) -> None:
    """Meldet Zeichen, die nach der Normalisierung nur in Modellausgaben stehen.

    Kein Fehler an sich – ein falsch gelesenes '%' statt 'i.' gehört genau
    hierhin. Aber neues Gerüst eines neuen Modells (Tags, Klammern, Pipes)
    fällt so auf, statt unbemerkt die CER zu erhöhen."""
    gold_zeichen = set().union(*(set(t) for t in gold_texte.values()))
    treffer: dict[str, list[str]] = {}
    for name, text in hyp_texte.items():
        for z in sorted(set(text) - gold_zeichen):
            treffer.setdefault(z, []).append(name)
    if not treffer:
        print("Artefakt-Kontrolle: keine goldfremden Zeichen in den "
              "normalisierten Ausgaben.")
        return
    print("Artefakt-Kontrolle – Zeichen nur in Modellausgaben "
          "(prüfen: Lesefehler oder übersehenes Gerüst?):")
    for z, namen in sorted(treffer.items()):
        print(f"  {z!r} U+{ord(z):04X} {unicodedata.name(z, '?')}  in {len(namen)}x: "
              f"{', '.join(namen[:3])}{' …' if len(namen) > 3 else ''}")


# --- Hauptlauf ----------------------------------------------------------------

def main() -> None:
    txt_paths = sorted(TRANSCRIPTIONS_DIR.glob("*.txt"))
    if not txt_paths:
        print(f"Keine Transkriptionen gefunden in {TRANSCRIPTIONS_DIR}")
        return

    rows = []
    gold_texte: dict[str, str] = {}
    hyp_texte: dict[str, str] = {}

    for txt_path in txt_paths:
        image_stem, model_slug = parse_filename(txt_path)
        gold_path = GOLD_DIR / f"{image_stem}.txt"
        if not gold_path.exists():
            print(f"  ! Kein Gold-Transkript für '{image_stem}' -> {txt_path.name} übersprungen")
            continue

        # .strip(): der abschließende Zeilenumbruch der Golddateien ist eine
        # Eigenschaft des Dateiformats, keine Auslassung des Modells.
        gold_raw = unicodedata.normalize("NFC", gold_path.read_text(encoding=ENCODING)).strip()
        hyp_raw = unicodedata.normalize("NFC", txt_path.read_text(encoding=ENCODING)).strip()

        gold_norm = normalize(gold_raw)
        hyp_norm = normalize(hyp_raw)
        wiederholungen = 1
        if DEDUPLICATE:
            hyp_norm, wiederholungen = deduplicate(hyp_norm)

        # Wie viel der Ausgabe war reines Gerüst? -> Maß für Nachbearbeitungsbedarf
        hyp_kompakt = kompakt(hyp_raw)
        artefakt_zeichen = len(hyp_kompakt) - len(hyp_norm)
        artefakt_anteil = round(artefakt_zeichen / len(hyp_kompakt), 4) if hyp_kompakt else 0.0

        # Normalisierte Texte sichern -> nachvollziehbar, was tatsächlich gemessen wurde
        norm_hyp_path = NORMALIZED_DIR / f"{txt_path.stem}.txt"
        norm_hyp_path.write_text(hyp_norm, encoding=ENCODING)
        gold_norm_path = NORMALIZED_DIR / f"GOLD__{image_stem}.txt"
        gold_norm_path.write_text(gold_norm, encoding=ENCODING)
        gold_texte[image_stem] = gold_norm
        hyp_texte[txt_path.stem] = hyp_norm

        for level, (g, h) in {"roh": (gold_raw, hyp_raw),
                              "norm": (gold_norm, hyp_norm)}.items():
            rows.append({
                "model": model_slug,
                "image": image_stem,
                "stufe": level,
                **metrics(g, h),
                "wiederholungen": wiederholungen,
                "artefakt_zeichen": artefakt_zeichen,
                "artefakt_anteil": artefakt_anteil,
            })

        if SAVE_HTML_DIFF_REPORTS:
            save_html_diff_report(gold_norm_path, norm_hyp_path, txt_path.stem)

        flag = f"  [{wiederholungen}x wiederholt]" if wiederholungen > 1 else ""
        roh, norm = rows[-2], rows[-1]
        print(f"{model_slug:30s} {image_stem:12s} "
              f"roh CER {roh['cer_dinglehopper']:.4f} WER {roh['wer_dinglehopper']:.4f}  ->  "
              f"norm CER {norm['cer_dinglehopper']:.4f} WER {norm['wer_dinglehopper']:.4f}"
              f"  (Gerüst {artefakt_anteil:.0%}){flag}")

    if not rows:
        print("Keine Zeile berechnet (fehlende Gold-Dateien?).")
        return

    df = pd.DataFrame(rows)
    out_csv = TABELLEN_DIR / "cer_wer_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} Zeilen gespeichert in {out_csv}")

    print("\n--- Vergleich (dinglehopper), sortiert nach CER norm ---")
    pivot = df.pivot_table(index="model", columns="stufe",
                           values=["cer_dinglehopper", "wer_dinglehopper"])
    print(pivot.sort_values(("cer_dinglehopper", "norm")).round(4))

    print("\n--- Anteil Gerüst an der Rohausgabe (Nachbearbeitungsbedarf) ---")
    print(df[df.stufe == "norm"].pivot_table(index="model", values="artefakt_anteil")
            .sort_values("artefakt_anteil", ascending=False).round(4))

    print()
    artefakt_kontrolle(gold_texte, hyp_texte)


if __name__ == "__main__":
    main()
