"""
Erzeugt aus den normalisierten Transkripten zwei Auswertungen:

  1. output/metrics/fehlerliste.csv
     Eine Zeile pro zusammenhaengendem Fehler, mit Kontext und
     Zeichenoperationen. Die Spalte 'typ' ist LEER und wird von Hand mit
     A-E gefuellt. Das ist die Arbeitsgrundlage fuer die Fehlertypologie.

  2. output/metrics/konfusionsmatrix.csv
     Zeichenverwechslungen (gold -> hyp) mit Haeufigkeit, gesamt und je
     Modell. '∅' steht fuer Einfuegung bzw. Loeschung.

Grundlage sind die NORMALISIERTEN Texte, damit Markdown-Artefakte nicht
als Zeichenverwechslungen in der Matrix landen.

Zwei bewusste Entscheidungen:

- Wortsegmentierung ueber .split() (nur an Leerzeichen), NICHT ueber
  dinglehopper.words(). dinglehopper trennt zusaetzlich an Bindestrichen
  und verwirft reine Interpunktions-Token; Fehler wie
  "Muehlen-Neu-" -> "Muehlen - Neu-" oder ":--:" -> "::" waeren damit
  unsichtbar - also ausgerechnet Typ B. Fuer die WER-Metrik in
  evaluate_ocr.py bleibt dinglehopper zustaendig, hier zaehlt
  Vollstaendigkeit.

- Aufeinanderfolgende Abweichungen werden zu EINER Zeile zusammengefasst.
  Sonst zerfaellt ein Fehler wie "Dresden" -> "Dres den" in mehrere
  Zeilen und ist von Hand kaum noch zuzuordnen.

Am Ende laeuft eine Probe: aus Gold + Fehlerliste wird die Modellausgabe
rekonstruiert. Stimmt sie nicht ueberein, fehlt der Liste etwas.
"""

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from dinglehopper import seq_align

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "output" / "metrics"
NORMALIZED_DIR = METRICS_DIR / "normalized"
ENCODING = "utf-8"

TYPOLOGIE = {
    "A": "graphemisch (ſ/f/s, Ligaturen ch/ck/tz, k/t, n/u)",
    "B": "typografisch (Sperrsatz, Interpunktion, Trennung, Lesereihenfolge)",
    "C": "orthographisch-historisch (stille Modernisierung: Telephon->Telefon)",
    "D": "lexikalisch (litauische Eigennamen, Diakritika, Lehnwoerter)",
    "E": "generative Artefakte (Halluzination, Wiederholung, Auslassung)",
}


def zeichen_ops(gold: str, hyp: str) -> str:
    """Zeichenunterschiede innerhalb eines Fehlerabschnitts."""
    if not gold:
        return f"eingefuegt: {hyp}"
    if not hyp:
        return f"geloescht: {gold}"
    ops = [f"{g or '∅'}->{h or '∅'}"
           for g, h in seq_align(list(gold), list(hyp)) if g != h]
    return " ".join(ops) if ops else "(identisch)"


def fehlerabschnitte(gold_tokens: list[str], hyp_tokens: list[str]):
    """Alignment durchlaufen und verschraenkte Abweichungen buendeln.

    Zwei benachbarte Abweichungen kommen nur dann in EINE Zeile, wenn
    mindestens eine davon eine Einfuegung oder Loeschung ist - dann ist die
    Zuordnung ohne den Nachbarn nicht lesbar ("Dresden" -> "Dres den").
    Zwei saubere Substitutionen nebeneinander bleiben getrennt, sonst
    landen zwei verschiedene Fehlertypen in derselben Zeile.

    Liefert (gold_text, hyp_text) je Abschnitt sowie die rekonstruierte
    Hypothese zur Kontrolle.
    """
    paare = list(seq_align(gold_tokens, hyp_tokens))
    rekon = [h for _, h in paare if h is not None]

    abschnitte, puffer = [], []

    def leeren():
        if puffer:
            abschnitte.append((" ".join(g for g, _ in puffer if g is not None),
                               " ".join(h for _, h in puffer if h is not None)))
            puffer.clear()

    def sauber(paar):
        """Plausible Wort-fuer-Wort-Substitution: beide Seiten vorhanden und
        einander aehnlich. Bildet das Alignment dagegen ein Wort auf etwas
        voellig anderes ab (z.B. 'Bedarfsartikel' -> '-'), ist es
        verschraenkt und darf nicht als eigene Zeile herausgeloest werden."""
        g, h = paar
        if g is None or h is None:
            return False
        return SequenceMatcher(None, g, h).ratio() >= 0.5

    for paar in paare:
        if paar[0] == paar[1]:
            leeren()
            continue
        if puffer and sauber(puffer[-1]) and sauber(paar):
            leeren()
        puffer.append(paar)
    leeren()
    return abschnitte, rekon


def main() -> None:
    gold_paths = sorted(NORMALIZED_DIR.glob("GOLD__*.txt"))
    if not gold_paths:
        print(f"Keine normalisierten Golddateien in {NORMALIZED_DIR}")
        print("Erst src/evaluate_ocr.py laufen lassen.")
        return

    fehler_rows, konf_rows, proben = [], [], {}

    for gold_path in gold_paths:
        image_stem = gold_path.stem.removeprefix("GOLD__")
        gold_text = gold_path.read_text(encoding=ENCODING)
        gold_tokens = gold_text.split()

        for hyp_path in sorted(NORMALIZED_DIR.glob(f"{image_stem}__*.txt")):
            model = hyp_path.stem.split("__", 1)[1]
            hyp_text = hyp_path.read_text(encoding=ENCODING)

            abschnitte, rekon = fehlerabschnitte(gold_tokens, hyp_text.split())
            proben[(image_stem, model)] = rekon

            for g, h in abschnitte:
                fehler_rows.append({
                    "image": image_stem,
                    "model": model,
                    "gold": g,
                    "hypothese": h,
                    "art": "Loeschung" if not h else
                           "Einfuegung" if not g else "Substitution",
                    "zeichen_ops": zeichen_ops(g, h),
                    "typ": "",          # <- von Hand: A/B/C/D/E
                    "bemerkung": "",    # <- optional
                })

            for g, h in seq_align(list(gold_text), list(hyp_text)):
                if g != h:
                    konf_rows.append({
                        "image": image_stem,
                        "model": model,
                        "gold_zeichen": g if g is not None else "∅",
                        "hyp_zeichen": h if h is not None else "∅",
                    })

    if not fehler_rows:
        print("Keine Fehler gefunden.")
        return

    luecken = [f"{img}/{m}" for (img, m), rek in proben.items()
               if rek != (NORMALIZED_DIR / f"{img}__{m}.txt")
               .read_text(encoding=ENCODING).split()]
    if luecken:
        print("! LUECKE in der Fehlerliste bei: " + ", ".join(luecken))
    else:
        print(f"Probe: alle {len(proben)} Modellausgaben aus Gold + "
              f"Fehlerliste rekonstruierbar -> vollstaendig\n")

    df_f = pd.DataFrame(fehler_rows)
    df_f.to_csv(METRICS_DIR / "fehlerliste.csv", index=False)

    df_k = pd.DataFrame(konf_rows)
    matrix = (df_k.groupby(["gold_zeichen", "hyp_zeichen"])
                  .size().reset_index(name="gesamt")
                  .sort_values("gesamt", ascending=False))
    je_modell = (df_k.groupby(["gold_zeichen", "hyp_zeichen", "model"])
                     .size().unstack(fill_value=0).reset_index())
    matrix = matrix.merge(je_modell, on=["gold_zeichen", "hyp_zeichen"])
    matrix.to_csv(METRICS_DIR / "konfusionsmatrix.csv", index=False)

    print(f"{len(df_f)} Fehler        -> {METRICS_DIR / 'fehlerliste.csv'}")
    print(f"{len(matrix)} Verwechslungen -> {METRICS_DIR / 'konfusionsmatrix.csv'}")
    print("\nTypologie zum Ausfuellen der Spalte 'typ':")
    for k, v in TYPOLOGIE.items():
        print(f"  {k}  {v}")


if __name__ == "__main__":
    main()