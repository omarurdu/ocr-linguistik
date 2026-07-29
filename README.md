# Fraktur-OCR mit LLMs

Untersuchung, wie gut multimodale LLMs (lokal via Ollama sowie Frontier-APIs) Fraktur-Text aus historischen Werbeanzeigen der *Litauischen Rundschau* transkribieren. Die Modell-Outputs werden gegen einen Goldstandard mit CER/WER gemessen und die Fehler anschließend in eine Typologie (A–E) einsortiert.

## Pipeline

```
data/images/*.png            Scans der Werbeanzeigen
data/gold/*.txt               Goldstandard-Transkriptionen (identischer Dateiname wie das Bild)
        │
        ▼  src/ollama_ocr.py         (lokale Modelle)
        ▼  src/frontier_ocr.py       (API-Modelle, WIP)
output/transcriptions/{bild}__{modell}.txt
        │
        ▼  src/evaluate_ocr.py
output/metrics/cer_wer_results.csv   CER/WER je Modell, roh vs. normalisiert
output/metrics/normalized/           normalisierte Texte (Kontrolle)
output/metrics/dinglehopper/         HTML-Diffs
        │
        ▼  src/fehleranalyse.py
output/metrics/fehlerliste.csv       ein Fehlerabschnitt pro Zeile, Spalte "typ" manuell auszufüllen
output/metrics/konfusionsmatrix.csv  Zeichenverwechslungen gold -> hyp
```

### Skripte

- **`src/ollama_ocr.py`** — schickt jedes Bild aus `data/images/` an jedes lokale Modell in `MODELS` (`ollama.chat`, `temperature=0`), entlädt jedes Modell danach explizit aus dem RAM, bevor das nächste geladen wird. Speichert Transkriptionen unter `output/transcriptions/` sowie eine Laufzeit-Übersicht in `output/ocr_results.csv`.
- **`src/frontier_ocr.py`** — Pendant für API-Modelle (OpenAI/Anthropic/Gemini), noch nicht implementiert.
- **`src/evaluate_ocr.py`** — berechnet CER/WER (jiwer + dinglehopper) auf zwei Stufen: *roh* (Modellausgabe wie geliefert) und *norm* (Markdown-/Layout-Artefakte deterministisch entfernt, identisch auf Gold und Hypothese angewandt). Historische Schreibweisen (langes ſ, Ligaturen, Groß-/Kleinschreibung) werden bewusst **nicht** normalisiert — das ist der Messgegenstand.
- **`src/fehleranalyse.py`** — baut auf den normalisierten Texten auf. Fasst zusammenhängende Abweichungen zu einer Fehlerzeile zusammen (Fehlerliste) und aggregiert Zeichenverwechslungen (Konfusionsmatrix). Prüft am Ende per Rekonstruktion, ob die Fehlerliste vollständig ist.

Fehlertypologie (`typ`-Spalte in `fehlerliste.csv`, manuell zu vergeben):

| Typ | Bedeutung |
|---|---|
| A | Graphemisch (ſ/f/s, Ligaturen ch/ck/tz, k/t, n/u) |
| B | Typografisch (Sperrsatz, Interpunktion, Trennung, Lesereihenfolge) |
| C | Orthographisch-historisch (stille Modernisierung, z. B. Thür→Tür) |
| D | Lexikalisch (litauische Eigennamen, Diakritika, Lehnwörter) |
| E | Generative Artefakte (Halluzination, Wiederholung, Auslassung) |

## Setup

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

API-Keys für die Frontier-Modelle in `.env` ablegen (Vorlage: `.env.template`):

```bash
cp .env.template .env
```

Für die lokalen Modelle wird eine laufende [Ollama](https://ollama.com)-Instanz mit den in `src/ollama_ocr.py` gelisteten Modellen benötigt.

### Ollama installieren

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

(Alternativ: Installer für macOS/Windows von [ollama.com/download](https://ollama.com/download).) Danach läuft Ollama als lokaler Dienst auf Port `11434`.

### Modelle pullen

```bash
ollama pull qwen3-vl:4b
ollama pull qwen3-vl:2b
ollama pull maternion/lightonocr-2:1b
ollama pull glm-ocr:latest
ollama pull deepseek-ocr:3b
ollama pull gemma4:e4b
```

Die Liste muss mit `MODELS` in [`src/ollama_ocr.py`](src/ollama_ocr.py:9) übereinstimmen — wird dort ein Modell hinzugefügt oder entfernt, hier entsprechend nachziehen.

## Verwendung

```bash
python src/ollama_ocr.py       # OCR mit lokalen Modellen
python src/evaluate_ocr.py     # CER/WER berechnen
python src/fehleranalyse.py    # Fehlerliste + Konfusionsmatrix erzeugen
```

Reihenfolge ist verbindlich: `fehleranalyse.py` benötigt die normalisierten Texte aus `evaluate_ocr.py`.

## Projektstruktur

```
data/
  images/         Scans der Werbeanzeigen (PNG)
  gold/           Goldstandard-Transkriptionen (TXT)
src/
  ollama_ocr.py       OCR über lokale Ollama-Modelle
  frontier_ocr.py      OCR über API-Modelle (WIP)
  evaluate_ocr.py       CER/WER-Berechnung
  fehleranalyse.py      Fehlerliste + Konfusionsmatrix
output/
  transcriptions/       Modell-Transkriptionen
  metrics/               CER/WER-Ergebnisse, Fehlerliste, Konfusionsmatrix
```

`output/` und `hausarbeit/` sind gitignored (lokale Zwischenergebnisse bzw. Arbeitsnotizen zur Hausarbeit).
