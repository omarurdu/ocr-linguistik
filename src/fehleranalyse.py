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

Drei bewusste Entscheidungen:

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

- 'typ' bleibt Handarbeit. Das Skript liefert daneben einen VORSCHLAG
  ('typ_vorschlag') samt der Regel, die ihn erzeugt hat ('regel'). Nur
  mechanisch entscheidbare Faelle bekommen einen Vorschlag; alles
  Interpretationsbeduerftige bleibt leer. Beispiel fuer die noetige
  Nachkontrolle: die Regel 'nur-diakritika' schlaegt D vor, aber
  KOENIGSBERG->KONIGSBERG ist eher ein graphemischer Fehler (A) als ein
  lexikalischer. Der Vorschlag ist eine Vorsortierung, kein Ergebnis.

Zusaetzliche Spalten der Fehlerliste, die aus der Auswertung der ersten
Modelle entstanden sind:

  gold_pos      Position im Gold-Tokenstrom - macht Haeufungen sichtbar
                (deepseek verliert z.B. bevorzugt Display-Zeilen am Anfang).
  kontext       Gold-Umgebung des Fehlers. Ohne sie ist Typ C (stille
                Modernisierung) von Typ A kaum zu unterscheiden.
  n_modelle     Wie viele Modelle genau diesen Fehler machen. Trennt
                Materialschwierigkeit von Modellschwaeche: 'Oele.'->'Dele.'
                bei 5 von 8 Modellen sagt etwas ueber die Fraktur-Vorlage,
                nicht ueber ein einzelnes Modell.
  verschiebung  Klammert Loeschung und Einfuegung desselben Textes
                zusammen. 'ECEKA' steht in anzeige04 oben, mehrere Modelle
                setzen es ans Ende - das ist EIN Lesereihenfolge-Fehler
                (Typ B) und nicht zwei unabhaengige.

Am Ende laeuft eine Probe: aus Gold + Fehlerliste wird die Modellausgabe
rekonstruiert. Stimmt sie nicht ueberein, fehlt der Liste etwas.
"""

import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from dinglehopper import seq_align

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "output" / "metrics"
NORMALIZED_DIR = METRICS_DIR / "normalized"
ENCODING = "utf-8"
KONTEXT_TOKEN = 3                 # Gold-Woerter links und rechts des Fehlers
VERSCHIEBUNG_AEHNLICHKEIT = 0.9   # ab wann gilt ein Block als nur verschoben
MIN_VERSCHIEBUNG_ZEICHEN = 4      # kuerzere Fragmente paaren zu leicht zufaellig

TYPOLOGIE = {
    "A": "graphemisch (ſ/f/s, Ligaturen ch/ck/tz, k/t, n/u)",
    "B": "typografisch (Sperrsatz, Interpunktion, Trennung, Lesereihenfolge)",
    "C": "orthographisch-historisch (stille Modernisierung: Telephon->Telefon)",
    "D": "lexikalisch (litauische Eigennamen, Diakritika, Lehnwoerter)",
    "E": "generative Artefakte (Halluzination, Wiederholung, Auslassung)",
}


# --- Vorklassifikation --------------------------------------------------------
# Bewusst konservativ: eine Regel feuert nur, wenn der Fall mechanisch
# entscheidbar ist. Alles andere bleibt leer und wird von Hand getippt.

# Beobachtete Frakturverwechslungen aus den ersten acht Modellen. Ungerichtet,
# weil beide Richtungen vorkommen (Werk->Wert und Wert->Werk).
FRAKTUR_PAARE = {
    frozenset("sf"),    # langes ſ als f gelesen: Wasser->Waffer, Essenzen->Effenzen
    frozenset("ſf"), frozenset("ſs"),
    frozenset("kt"),    # Werk->Wert, Lokomobilen->Lotomobilen
    frozenset("kf"),    # Artikel->Artifel, Fabrik->Fabrif
    frozenset("kl"),    # Verkauf->Verlauf
    frozenset("tf"),    # Charlottenburg->Charloffenburg
    frozenset("AU"),    # Anfertigung->Unfertigung, Amol->Umol
    frozenset("OD"),    # Oele->Dele
    frozenset("xr"),    # Export->Erport
    frozenset("nu"), frozenset("ce"), frozenset("mw"),
    frozenset("IJ"), frozenset("il"), frozenset("bh"), frozenset("vy"),
}

# Zeichen, an denen sich Segmentierung und Trennung entscheiden
TRENNZEICHEN = set(" -‐‑–—")


def ohne_diakritika(text: str) -> str:
    """Zerlegt und wirft kombinierende Zeichen weg. 'Šiaulių' -> 'Siauliu'.
    ß bleibt stehen (nicht zerlegbar) - 'Grössen'/'Größen' ist damit
    korrekterweise KEIN Diakritikafall, sondern Typ C."""
    zerlegt = unicodedata.normalize("NFD", text)
    return "".join(z for z in zerlegt if not unicodedata.combining(z))


def ohne_trennung(text: str) -> str:
    """Entfernt Leer- und Trennzeichen: 'ge- hörigen' und 'gehörigen' werden
    gleich, 'E C E K A' und 'ECEKA' ebenfalls."""
    return "".join(z for z in text if z not in TRENNZEICHEN)


def nur_frakturverwechslung(gold: str, hyp: str) -> bool:
    """Wahr, wenn sich beide Seiten ausschliesslich in bekannten Fraktur-
    verwechslungen unterscheiden. Verlangt gleiche Laenge - sobald Zeichen
    dazukommen oder wegfallen (Ligaturen wie tz->k), ist der Fall nicht mehr
    mechanisch sicher und geht an die Handarbeit."""
    if len(gold) != len(hyp) or gold == hyp:
        return False
    unterschiede = [(g, h) for g, h in zip(gold, hyp) if g != h]
    return all(frozenset((g, h)) in FRAKTUR_PAARE for g, h in unterschiede)


def klassifiziere(gold: str, hyp: str, art: str, verschoben: bool) -> tuple[str, str]:
    """Liefert (typ_vorschlag, regel). Leerer Vorschlag = Handarbeit noetig."""
    if verschoben:
        return "B", "verschiebung"
    if art in ("Loeschung", "Einfuegung"):
        # Nur wortfoermiges Material gilt als Auslassung/Halluzination.
        # Verlorene Interpunktion ist typografisch, nicht generativ.
        text = gold or hyp
        return ("E", "block-ausgelassen" if art == "Loeschung" else "block-erfunden") \
            if any(z.isalpha() for z in text) else ("", "")
    if gold != hyp and ohne_diakritika(gold) == ohne_diakritika(hyp):
        return "D", "nur-diakritika"
    if gold != hyp and ohne_trennung(gold) == ohne_trennung(hyp):
        return "B", "nur-trennung"
    if nur_frakturverwechslung(gold, hyp):
        return "A", "nur-fraktur"
    return "", ""


def finde_verschiebungen(rows: list[dict]) -> None:
    """Paart Loeschungen mit der aehnlichsten Einfuegung derselben Ausgabe und
    traegt bei beiden dieselbe Kennung ein.

    Verglichen wird ohne Leer- und Trennzeichen, damit gesperrt gesetztes
    'E C E K A' auf 'ECEKA' passt. Verlangt wird keine Gleichheit, sondern
    AEHNLICHKEIT: minicpm setzt in anzeige01 die Logo-Inschrift ans Ende und
    laesst dabei die Anfuehrungszeichen weg ('FABRIKO ŽENKLAS „AUŠRA“' ->
    'FABRIKO ŽENKLAS AUŠRA'). Bei Gleichheitspruefung bliebe das eine
    Auslassung PLUS eine Halluzination - beides falsch, der Block ist nur
    verschoben. Der Rest-Unterschied bleibt in den Zeilen sichtbar.

    Aendert 'rows' an Ort und Stelle. Beide Zeilen bleiben erhalten - wuerde
    man sie zusammenfassen, liesse sich die Modellausgabe nicht mehr aus der
    Fehlerliste rekonstruieren (siehe Probe am Ende)."""
    offen = [r for r in rows if r["art"] == "Einfuegung"]
    for nr, weg in enumerate([r for r in rows if r["art"] == "Loeschung"], start=1):
        schluessel = ohne_trennung(weg["gold"])
        if len(schluessel) < MIN_VERSCHIEBUNG_ZEICHEN:
            continue
        bewertet = [(SequenceMatcher(None, schluessel,
                                     ohne_trennung(e["hypothese"])).ratio(), e)
                    for e in offen]
        bewertet = [(q, e) for q, e in bewertet if q >= VERSCHIEBUNG_AEHNLICHKEIT]
        if not bewertet:
            continue
        treffer = max(bewertet, key=lambda p: p[0])[1]
        offen.remove(treffer)
        kennung = f"{weg['image']}/{weg['model']}#{nr}"
        weg["verschiebung"] = treffer["verschiebung"] = kennung


# --- Fehlerabschnitte ---------------------------------------------------------

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

    Liefert je Abschnitt (position_im_gold, gold_text, hyp_text).
    """
    paare = list(seq_align(gold_tokens, hyp_tokens))

    abschnitte, puffer = [], []
    gold_i = 0      # naechste unverbrauchte Position im Gold
    start = 0       # Gold-Position, an der der aktuelle Puffer beginnt

    def leeren():
        if puffer:
            abschnitte.append((start,
                               " ".join(g for g, _ in puffer if g is not None),
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
            gold_i += 1
            continue
        if puffer and sauber(puffer[-1]) and sauber(paar):
            leeren()
        if not puffer:
            start = gold_i
        puffer.append(paar)
        if paar[0] is not None:
            gold_i += 1
    leeren()
    return abschnitte


def rekonstruiere(gold_tokens: list[str], abschnitte: list[tuple]) -> list[str]:
    """Baut die Modellausgabe aus Gold + Fehlerabschnitten nach. Nur wenn das
    Ergebnis exakt der Modellausgabe entspricht, ist die Fehlerliste
    vollstaendig - eine uebersehene Abweichung faellt hier sofort auf."""
    ausgabe, i = [], 0
    for start, gold, hyp in abschnitte:
        ausgabe.extend(gold_tokens[i:start])
        if hyp:
            ausgabe.extend(hyp.split())
        i = start + len(gold.split())
    ausgabe.extend(gold_tokens[i:])
    return ausgabe


def kontext(gold_tokens: list[str], start: int, laenge: int) -> str:
    """Gold-Umgebung des Fehlers, Fundstelle in [[...]] markiert."""
    links = " ".join(gold_tokens[max(0, start - KONTEXT_TOKEN):start])
    mitte = " ".join(gold_tokens[start:start + laenge])
    rechts = " ".join(gold_tokens[start + laenge:start + laenge + KONTEXT_TOKEN])
    return " ".join(t for t in (links, f"[[{mitte or '∅'}]]", rechts) if t)


# --- Hauptlauf ----------------------------------------------------------------

def main() -> None:
    gold_paths = sorted(NORMALIZED_DIR.glob("GOLD__*.txt"))
    if not gold_paths:
        print(f"Keine normalisierten Golddateien in {NORMALIZED_DIR}")
        print("Erst src/evaluate_ocr.py laufen lassen.")
        return

    fehler_rows, konf_rows, luecken = [], [], []
    gelegenheiten: dict[str, int] = {}
    geprueft = 0

    for gold_path in gold_paths:
        image_stem = gold_path.stem.removeprefix("GOLD__")
        gold_text = gold_path.read_text(encoding=ENCODING)
        gold_tokens = gold_text.split()
        hyp_paths = sorted(NORMALIZED_DIR.glob(f"{image_stem}__*.txt"))

        for hyp_path in hyp_paths:
            model = hyp_path.stem.split("__", 1)[1]
            hyp_text = hyp_path.read_text(encoding=ENCODING)

            abschnitte = fehlerabschnitte(gold_tokens, hyp_text.split())
            geprueft += 1
            if rekonstruiere(gold_tokens, abschnitte) != hyp_text.split():
                luecken.append(f"{image_stem}/{model}")

            modell_rows = []
            for start, g, h in abschnitte:
                modell_rows.append({
                    "image": image_stem,
                    "model": model,
                    "gold_pos": start,
                    "gold": g,
                    "hypothese": h,
                    "art": "Loeschung" if not h else
                           "Einfuegung" if not g else "Substitution",
                    "kontext": kontext(gold_tokens, start, len(g.split())),
                    "zeichen_ops": zeichen_ops(g, h),
                    "verschiebung": "",
                })
            # Bezugsgroesse fuer die Matrix: welche Goldzeichen standen bei
            # DIESEM Vergleich ueberhaupt zur Zeichenverwechslung an? Woerter
            # aus komplett ausgefallenen Bloecken zaehlen nicht mit - sie hatten
            # nie eine Hypothese gegenueber, an der sie haetten scheitern
            # koennen. Ueber alle Modelle summiert ergibt das die Zahl der
            # Gelegenheiten; 'anteil' = 0.25 heisst dann: jedes vierte Mal, das
            # dieses Zeichen zu lesen war, wurde es so verwechselt.
            ausgefallen = {i for start, g, h in abschnitte if not h
                           for i in range(start, start + len(g.split()))}
            for z, n in Counter(" ".join(t for i, t in enumerate(gold_tokens)
                                         if i not in ausgefallen)).items():
                gelegenheiten[z] = gelegenheiten.get(z, 0) + n

            finde_verschiebungen(modell_rows)
            for r in modell_rows:
                r["typ_vorschlag"], r["regel"] = klassifiziere(
                    r["gold"], r["hypothese"], r["art"], bool(r["verschiebung"]))
                r["typ"] = ""          # <- von Hand: A/B/C/D/E
                r["bemerkung"] = ""    # <- optional
            fehler_rows.extend(modell_rows)

            # Die Matrix entsteht NUR aus Substitutionsabschnitten, also dort,
            # wo ein Goldwort tatsaechlich auf ein Hypothesenwort abgebildet
            # wurde. Eine Zeichenalignierung ueber den ganzen Text waere
            # unbrauchbar: verschobene Bloecke ('FABRIKO ŽENKLAS AUŠRA',
            # 'ECEKA') zerfallen dabei in Dutzende Einzelloeschungen und
            # -einfuegungen. In einer frueheren Fassung stammten so 219 von 653
            # 'graphemischen' Faellen aus genau zwei verschobenen Bloecken - die
            # haeufigsten 'Verwechslungen' ergaben aneinandergereiht deren
            # Buchstabeninventar. Ganze Bloecke stehen als je eine Zeile in der
            # Fehlerliste; in eine ZEICHEN-Verwechslungsmatrix gehoeren sie nicht.
            for _, g, h in abschnitte:
                if not g or not h:
                    continue
                for a, b in seq_align(list(g), list(h)):
                    if a != b:
                        konf_rows.append({
                            "image": image_stem,
                            "model": model,
                            "gold_zeichen": a if a is not None else "∅",
                            "hyp_zeichen": b if b is not None else "∅",
                            # Leerzeichen-Verwechslungen sind Segmentierung, kein
                            # Leseproblem. Getrennt ausweisen, sonst ueberdecken
                            # sie in der Matrix die graphemischen Faelle.
                            "ebene": "Segmentierung" if " " in (a or "") + (b or "")
                                     else "graphemisch",
                        })

    if not fehler_rows:
        print("Keine Fehler gefunden.")
        return

    if luecken:
        print("! LUECKE in der Fehlerliste bei: " + ", ".join(luecken))
    else:
        print(f"Probe: alle {geprueft} Modellausgaben aus Gold + Fehlerliste "
              f"rekonstruierbar -> vollstaendig\n")

    df_f = pd.DataFrame(fehler_rows)

    # Wie viele Modelle machen exakt diesen Fehler? -> Materialschwierigkeit
    # gegen Modellschwaeche
    df_f["n_modelle"] = df_f.groupby(["image", "gold", "hypothese"])["model"] \
                            .transform("nunique")

    spalten = ["image", "model", "gold_pos", "gold", "hypothese", "art",
               "kontext", "zeichen_ops", "n_modelle", "verschiebung",
               "typ_vorschlag", "regel", "typ", "bemerkung"]
    df_f = df_f[spalten].sort_values(["image", "model", "gold_pos"])
    df_f.to_csv(METRICS_DIR / "fehlerliste.csv", index=False)

    df_k = pd.DataFrame(konf_rows)
    matrix = (df_k.groupby(["gold_zeichen", "hyp_zeichen", "ebene"])
                  .size().reset_index(name="gesamt")
                  .sort_values("gesamt", ascending=False))
    # Absolute Zahlen taeuschen: 14x s->f ist erst aussagekraeftig im Verhaeltnis
    # dazu, wie oft 's' ueberhaupt gelesen werden musste. 'anteil' = 0.25 heisst:
    # jedes vierte Mal wurde dieses Zeichen so verwechselt.
    matrix["gelegenheiten"] = matrix.gold_zeichen.map(gelegenheiten)
    matrix["anteil"] = (matrix.gesamt / matrix.gelegenheiten).round(4)
    je_modell = (df_k.groupby(["gold_zeichen", "hyp_zeichen", "model"])
                     .size().unstack(fill_value=0).reset_index())
    matrix = matrix.merge(je_modell, on=["gold_zeichen", "hyp_zeichen"])
    matrix.to_csv(METRICS_DIR / "konfusionsmatrix.csv", index=False)

    print(f"{len(df_f)} Fehler        -> {METRICS_DIR / 'fehlerliste.csv'}")
    print(f"{len(matrix)} Verwechslungen -> {METRICS_DIR / 'konfusionsmatrix.csv'}")

    offen = (df_f.typ_vorschlag == "").sum()
    print(f"\nVorklassifikation: {len(df_f) - offen} von {len(df_f)} Zeilen haben "
          f"einen Vorschlag, {offen} bleiben Handarbeit.")
    print("Vorschlaege je Regel (bitte stichprobenartig gegenlesen):")
    for (typ, regel), n in df_f[df_f.typ_vorschlag != ""] \
            .groupby(["typ_vorschlag", "regel"]).size().items():
        print(f"  {typ}  {regel:18s} {n:4d}")

    print("\nTypologie zum Ausfuellen der Spalte 'typ':")
    for k, v in TYPOLOGIE.items():
        print(f"  {k}  {v}")


if __name__ == "__main__":
    main()
