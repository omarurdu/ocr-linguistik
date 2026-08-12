"""
prompts_base.py - zentrale Ablage aller OCR-Prompts.

Einzige Quelle fuer die Prompt-Texte von src/ollama_ocr.py und
src/frontier_ocr.py. Der Vergleich lokal vs. API haengt daran, dass beide
Skripte denselben Generalisten-Prompt schicken; solange der Text nur hier
steht, kann er zwischen den Skripten nicht auseinanderlaufen.

Konvention: Text und ID gehoeren zusammen und werden nie getrennt kopiert. Die
Skripte importieren die ID-Konstanten und holen den Text ueber
PROMPT_TEXTS[prompt_id]. Die Herkunft jedes Prompts steht in PROMPT_META und
gehoert so in den Methodenteil.

Wird ein Prompt-Text geaendert, sind alte Laeufe nicht mehr vergleichbar. Dann
eine NEUE ID vergeben (z.B. P-GEN-02), statt den Text unter der alten ID zu
ueberschreiben. frontier_ocr.py schreibt zusaetzlich prompt_sha256 ins CSV -
damit faellt eine stille Aenderung nachtraeglich auf.
"""

# ---------------------------------------------------------------------------
# IDs. Landen unveraendert in der Spalte prompt_id der Ergebnis-CSVs.
# ---------------------------------------------------------------------------

PROMPT_ID_GEN = "P-GEN-02"
PROMPT_ID_GEN_V1 = "P-GEN-01"   # abgeloest, nur noch fuer alte Ergebnis-CSVs
PROMPT_ID_LIGHTON = "P-OCR-LIGHTON-01"
PROMPT_ID_GLM = "P-OCR-GLM-01"
PROMPT_ID_DSK = "P-OCR-DSK-01"

# ---------------------------------------------------------------------------
# Texte: einheitlich innerhalb der Generalisten, Task-Praefixe laut Modelcard
# bei den OCR-Spezialisten.
# ---------------------------------------------------------------------------

PROMPT_GEN = (
    "This image shows a printed advertisement from a historical German-language "
    "periodical. Transcribe all of its text exactly as printed, keeping the line "
    "breaks of the print; render the long s (ſ) as a normal s. Read from top to "
    "bottom; where two elements stand side by side, transcribe the left one "
    "completely before the right one. Include text that belongs to logos, "
    "trademarks or stamps. Do not modernise the spelling, do not describe "
    "pictures, and do not add any comment or formatting."
)

# Vorgaenger. Bleibt im Register, damit die prompt_id aelterer Ergebnis-CSVs
# aufloesbar bleibt; wird an kein Modell mehr geschickt.
PROMPT_GEN_V1 = (
    "Transkribiere den gesamten sichtbaren Text auf diesem Bild, exakt und ohne "
    "Kommentar, ohne Formatierung oder Zusätze."
)

PROMPT_LIGHTON = ""
PROMPT_GLM = "Text Recognition:"
PROMPT_DSK = "Free OCR."

PROMPT_TEXTS = {
    PROMPT_ID_GEN: PROMPT_GEN,
    PROMPT_ID_GEN_V1: PROMPT_GEN_V1,
    PROMPT_ID_LIGHTON: PROMPT_LIGHTON,
    PROMPT_ID_GLM: PROMPT_GLM,
    PROMPT_ID_DSK: PROMPT_DSK,
}

# ---------------------------------------------------------------------------
# Herkunft je Prompt - Belegkette fuer den Methodenteil.
# ---------------------------------------------------------------------------

PROMPT_META = {
    PROMPT_ID_GEN: (
        "eigene Formulierung; einheitlich fuer alle generalistischen Modelle, "
        "lokal wie API. Loest P-GEN-01 ab. Englisch, weil Instruction-Following "
        "ueber das ganze Modellfeld (1B bis Frontier) robuster ist als in "
        "Deutsch. Die Zusaetze gegenueber P-GEN-01 halten ausschliesslich die "
        "Transkriptionskonvention des Goldstandards fest - Lesereihenfolge, "
        "Werbetext in Bildmarken, langes s, keine Modernisierung -, damit die "
        "Modelle nicht gegen eine ungenannte Konvention gemessen werden. Keine "
        "Anweisung nimmt auf ein einzelnes Testbild Bezug."
    ),
    PROMPT_ID_GEN_V1: (
        "eigene Formulierung; Entwicklungsstand bis 2026-08-12, deutsch. "
        "Abgeloest durch P-GEN-02; Ergebnisse in runs/ archiviert"
    ),
    PROMPT_ID_LIGHTON: (
        "Modelcard lightonai/LightOnOCR-2-1B: Beispiel uebergibt nur das Bild, "
        "keinen Text"
    ),
    PROMPT_ID_GLM: "Modelcard zai-org/GLM-OCR: 'Text Recognition:'",
    PROMPT_ID_DSK: (
        "Modelcard deepseek-ai/DeepSeek-OCR: '<image>\\nFree OCR.'; Bildtoken "
        "setzt Ollama selbst"
    ),
}

# Jede ID braucht Text UND Herkunft - sonst steht spaeter eine prompt_id im CSV,
# zu der die Belegkette fehlt.
_unbelegt = set(PROMPT_TEXTS) ^ set(PROMPT_META)
if _unbelegt:
    raise RuntimeError(
        f"PROMPT_TEXTS und PROMPT_META decken sich nicht: {sorted(_unbelegt)}"
    )
