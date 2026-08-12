#!/usr/bin/env python3
"""
Bildinventar für die Materialbeschreibung (Hausarbeit: VLM-OCR über Ollama).

Liest nur die Header-Metadaten aller Rasterbilder in einem Ordner und schreibt
sie als CSV. Pixeldaten werden nicht dekodiert -> auch bei vielen Dateien schnell.

Aufruf:  python src/inventar_bilder.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------- Einstellungen

# Pfade relativ zur Projektwurzel (eine Ebene ueber src/), damit der Aufrufort
# keine Rolle spielt. TABELLEN_DIR sammelt alle CSV/MD-Ausgaben des Projekts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BILD_DIR = PROJECT_ROOT / "data" / "images"
TABELLEN_DIR = PROJECT_ROOT / "output" / "tabellen"
OUT_CSV = TABELLEN_DIR / "bilder_inventar.csv"

ZIEL_KANTE = 1540          # LightOnOCR-2: empfohlene längste Bildkante in px
ENDUNGEN = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# True -> Semikolon als Trenner und Komma als Dezimalzeichen (deutsches Excel)
DEUTSCHES_EXCEL = True

MODUS_KLARTEXT = {
    "1":    "bitonal (1 Bit)",
    "L":    "Graustufen (8 Bit)",
    "LA":   "Graustufen + Alpha",
    "I;16": "Graustufen (16 Bit)",
    "P":    "Palette (indiziert)",
    "PA":   "Palette + Alpha",
    "RGB":  "Farbe (RGB)",
    "RGBA": "Farbe + Alpha",
    "CMYK": "CMYK",
}

# ---------------------------------------------------------------- Hilfsfunktionen


def dpi_werte(im: Image.Image) -> tuple:
    """Gibt (dpi_x, dpi_y) zurück, oder ('', '') wenn nicht im Header hinterlegt.

    Hinweis: PNG speichert die Auflösung im pHYs-Chunk in Pixel/Meter. Die
    Rueckrechnung liefert deshalb oft krumme Werte (z.B. 299.9994 statt 300).
    """
    dpi = im.info.get("dpi")
    if not dpi:
        return "", ""
    try:
        return round(float(dpi[0]), 1), round(float(dpi[1]), 1)
    except (TypeError, ValueError, IndexError):
        return "", ""


def bilddateien(ordner: Path) -> list:
    """Alle Bilddateien im Ordner und in Unterordnern, Endung case-insensitiv."""
    return sorted(
        (p for p in ordner.rglob("*") if p.suffix.lower() in ENDUNGEN),
        key=lambda p: p.name.lower(),
    )


def zeile_bauen(p: Path) -> dict:
    with Image.open(p) as im:
        w, h = im.width, im.height
        lang = max(w, h)
        dx, dy = dpi_werte(im)

        if lang < ZIEL_KANTE:
            skalierung = "hochskaliert"
        elif lang > ZIEL_KANTE:
            skalierung = "verkleinert"
        else:
            skalierung = "exakt"

        return {
            "datei": p.name,
            "format": im.format,
            "breite_px": w,
            "hoehe_px": h,
            "laengste_kante_px": lang,
            "megapixel": round(w * h / 1e6, 2),
            "seitenverhaeltnis": round(w / h, 3) if h else "",
            "modus": im.mode,
            "farbtiefe_klartext": MODUS_KLARTEXT.get(im.mode, im.mode),
            "dpi_x": dx,
            "dpi_y": dy,
            "dpi_quelle": "Dateiheader" if dx != "" else "nicht hinterlegt",
            "faktor_auf_1540": round(ZIEL_KANTE / lang, 2) if lang else "",
            "skalierung": skalierung,
            "dateigroesse_kb": round(p.stat().st_size / 1024, 1),
        }


def eindeutschen(zeilen: list) -> list:
    """Punkt -> Komma in Zahlenfeldern, damit Excel (DE) sie als Zahl liest."""
    raus = []
    for z in zeilen:
        raus.append({
            k: (str(v).replace(".", ",") if isinstance(v, float) else v)
            for k, v in z.items()
        })
    return raus


def auswertung_drucken(zeilen: list) -> None:
    kanten = [z["laengste_kante_px"] for z in zeilen]
    hoch = [z["datei"] for z in zeilen if z["skalierung"] == "hochskaliert"]
    modi = sorted({z["farbtiefe_klartext"] for z in zeilen})
    ohne_dpi = [z["datei"] for z in zeilen if z["dpi_quelle"] == "nicht hinterlegt"]

    print(f"\n{len(zeilen)} Bilder inventarisiert.")
    print(f"Längste Kante: {min(kanten)}–{max(kanten)} px (Ziel: {ZIEL_KANTE} px)")
    print(f"Farbtiefen im Korpus: {', '.join(modi)}")
    if hoch:
        print(f"ACHTUNG hochskaliert ({len(hoch)}): {', '.join(hoch)}")
        print("  -> Vergrößerung fügt keine Information hinzu. Im Kapitel Material erwähnen.")
    if ohne_dpi:
        print(f"Ohne DPI im Header ({len(ohne_dpi)}): {', '.join(ohne_dpi)}")
        print("  -> Effektive Auflösung ggf. per 'pdfimages -list quelle.pdf' aus dem PDF holen.")


# ---------------------------------------------------------------- Hauptprogramm


def main() -> int:
    if not BILD_DIR.is_dir():
        print(f"FEHLER: Ordner '{BILD_DIR}' existiert nicht.", file=sys.stderr)
        return 1

    pfade = bilddateien(BILD_DIR)
    if not pfade:
        print(f"FEHLER: keine Bilddateien in '{BILD_DIR}' gefunden.", file=sys.stderr)
        print(f"Gesucht wurde nach: {', '.join(sorted(ENDUNGEN))}", file=sys.stderr)
        return 1

    zeilen, fehler = [], []
    for p in pfade:
        try:
            zeilen.append(zeile_bauen(p))
        except (UnidentifiedImageError, OSError) as e:
            fehler.append((p.name, str(e)))

    for name, msg in fehler:
        print(f"ÜBERSPRUNGEN: {name} ({msg})", file=sys.stderr)

    if not zeilen:
        print("FEHLER: keine Datei konnte gelesen werden.", file=sys.stderr)
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ausgabe = eindeutschen(zeilen) if DEUTSCHES_EXCEL else zeilen
    trenner = ";" if DEUTSCHES_EXCEL else ","

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(zeilen[0].keys()), delimiter=trenner)
        w.writeheader()
        w.writerows(ausgabe)

    print(f"Geschrieben: {OUT_CSV}")
    auswertung_drucken(zeilen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())