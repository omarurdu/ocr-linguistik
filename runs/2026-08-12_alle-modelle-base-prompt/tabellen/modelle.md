# Modellinventar (lokal, Ollama)

- Erhebungsdatum: 2026-08-11
- Ollama-Version: ollama version is 0.31.2
- Host: tzuru-BOM-WXX9
- Rohdaten: `modelle.json`

## A — Automatisch ausgelesen

| Alias | Digest | Parameter | Quantisierung | GB | Capabilities | Kontext | Stand |
|---|---|---|---|---|---|---|---|
| qwen3-vl:4b-instruct | ee4b975b58c1 | 4.4B | Q4_K_M | 3.3 | completion, vision, tools | 262144 | 2026-08-11 |
| qwen3-vl:8b-instruct | 0533d74300e4 | 8.8B | Q4_K_M | 6.14 | completion, vision, tools | 262144 | 2026-08-11 |
| gemma4:e4b | c6eb396dbd59 | 8.0B | Q4_K_M | 9.61 | completion, vision, audio, tools, thinking | 131072 | 2026-07-21 |
| qwen3.5:4b | 2a654d98e6fb | 4.7B | Q4_K_M | 3.39 | completion, vision, tools, thinking | 262144 | 2026-08-10 |
| maternion/lightonocr-2:1b | ee7d83c4eb67 | 596.05M | Q8_0 | 1.47 | completion, vision, tools | 16384 | 2026-07-20 |
| glm-ocr:bf16 | 6effedd0dc8a | 1.1B | F16 | 2.22 | completion, vision, tools | 131072 | 2026-08-10 |
| deepseek-ocr:3b | 0e7b018b8a22 | 3.3B | F16 | 6.69 | completion, vision | 8192 | 2026-07-20 |
| minicpm-v4.5:8b | 0c40168f46d1 | 8.2B | Q4_K_M | 6.12 | tools, thinking, completion, vision | 40960 | 2026-07-29 |

## B — Von Hand nachzutragen

| Alias | Klasse | Upstream/Original | Lizenz | Referenz | Publikationsstatus | Prompt-ID |
|---|---|---|---|---|---|---|
| qwen3-vl:4b-instruct |  |  | Apache License |  |  |  |
| qwen3-vl:8b-instruct |  |  | Apache License |  |  |  |
| gemma4:e4b |  |  | Apache License |  |  |  |
| qwen3.5:4b |  |  | Apache License |  |  |  |
| maternion/lightonocr-2:1b |  |  |  |  |  |  |
| glm-ocr:bf16 |  |  |  |  |  |  |
| deepseek-ocr:3b |  |  | MIT License |  |  |  |
| minicpm-v4.5:8b |  |  |  |  |  |  |

## C — Laufparameter

Einheitlich fuer alle lokalen Modelle gesetzt (im Skript, nicht Ollama-Default):

| Parameter | Wert |
|---|---|
| temperature | 0 |
| seed | <eintragen> |
| top_p | <eintragen> |
| top_k | <eintragen> |
| num_ctx | <eintragen> |
| num_predict | <eintragen> |
| think | <eintragen, siehe unten> |

Ollama-Defaults je Modell (zur Kontrolle, ob die eigenen Settings greifen):

| Alias | Defaults laut /api/show |
|---|---|
| qwen3-vl:4b-instruct | temperature                    1; top_k                          20; top_p                          0.95 |
| qwen3-vl:8b-instruct | top_p                          0.95; temperature                    1; top_k                          20 |
| gemma4:e4b | top_p                          0.95; temperature                    1; top_k                          64 |
| qwen3.5:4b | presence_penalty               1.5; temperature                    1; top_k                          20; top_p                          0.95 |
| maternion/lightonocr-2:1b | temperature                    0.2; top_p                          0.9 |
| glm-ocr:bf16 | temperature                    0 |
| deepseek-ocr:3b | temperature                    0 |
| minicpm-v4.5:8b |  |

## D — Bildvorverarbeitung

| Feld | Wert |
|---|---|
| Quellformat / Farbtiefe | <eintragen> |
| Aufloesung Goldstandard-Ausschnitt (px) | <eintragen> |
| Skalierung vor Uebergabe | <eintragen> |
