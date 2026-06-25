# Favorita Store Sales Forecasting: Multi-Model Pipeline

Dieses Projekt beschäftigt sich mit der Vorhersage von Verkaufszahlen für die ecuadorianische Supermarktkette **Favorita**. Ziel ist es, die täglichen Transaktionen einzelner Stores mithilfe verschiedener Machine-Learning- und Deep-Learning-Modelle vorherzusagen.

Die zentrale Herausforderung besteht in der Modellierung komplexer Zeitreihen, die durch zahlreiche externe Faktoren beeinflusst werden. Dazu gehören unter anderem Feiertage, lokale Events, saisonale Muster, Wochentagseffekte sowie makroökonomische Faktoren wie der Ölpreis.

Statt ein einzelnes Modell zu verwenden, wurde eine skalierbare **Multi-Model-Forecasting-Pipeline** entwickelt, die unterschiedliche Modellklassen systematisch vergleicht und hinsichtlich ihrer Prognosefähigkeit evaluiert.

---

# 🛠 Methodik & Umsetzung

## 1. Daten-Architektur & Performance

Die Datenverarbeitung wurde auf eine effiziente und skalierbare Struktur ausgelegt:

* **High-Speed Processing:** Konsequenter Einsatz von **Polars** für Data Wrangling und Feature Engineering, um eine speichereffiziente und parallele Verarbeitung großer Zeitreihen zu ermöglichen.
* **Storage:** Verwendung des **Parquet-Formats** für komprimierte Speicherung und schnellen Zugriff auf Trainings-, Validierungs- und Testdaten.
* **Modularität:** Trennung der Kernlogik (`03_src/`) von Analyse- und Trainingsnotebooks (`04_notebooks/`), wodurch Wartbarkeit und Wiederverwendbarkeit verbessert werden.
* **Konfiguration:** Zentrale Verwaltung von Modellparametern und Pfaden über Konfigurationsdateien.

---

# 2. Multivariates Feature Engineering

Um den Modellen zusätzliche Informationen über zeitliche Muster und externe Einflussfaktoren bereitzustellen, wurden umfangreiche Features erzeugt.

## Zeitbasierte Features

* Wochentag
* Monat
* Jahr
* Kalenderwoche
* Feiertagsindikatoren
* Pre- und Post-Holiday-Fenster
* Saisonale Muster

Diese Features ermöglichen es den Modellen, wiederkehrende Nachfragezyklen und besondere Ereignisse abzubilden.

## Historische Zeitreihen-Features

Zur Modellierung von Dynamiken innerhalb der Verkaufszeitreihen wurden erzeugt:

* **Rolling Means**
  * 7 Tage
  * 14 Tage
  * 28 Tage

* **Lag Features**
  * vergangene Transaktionswerte
  * saisonale Verzögerungen
  * historische Vergleichszeiträume

* **Momentum Features**
  * Wachstumsraten
  * Veränderung gegenüber vergangenen Perioden

* **Difference Features**
  * absolute Differenzen zu historischen Referenzen

* **Ratio Features**
  * Verhältnis aktueller Werte zu historischen Benchmarks

## Externe Einflussfaktoren

Zusätzlich wurden externe Variablen integriert:

* täglicher Ölpreis als wirtschaftlicher Indikator
* nationale und regionale Feiertage
* lokale Events

---

# 3. Modellvergleich & Forecasting-Ansätze

Die Pipeline vergleicht drei unterschiedliche Modellfamilien.

## Gradient Boosted Trees

### XGBoost & LightGBM

Diese Modelle arbeiten auf einer klassischen Feature-Matrix und lernen nichtlineare Zusammenhänge zwischen Eingangsvariablen und Verkaufszahlen.

Vorteile:

* hohe Performance bei tabellarischen Daten
* robuste Verarbeitung vieler Features
* gute Interpretierbarkeit durch Feature Importance

---

## Deep Learning Modelle

### PatchTST & NHITS

Die neuronalen Modelle werden über `neuralforecast` trainiert und verarbeiten historische Sequenzen direkt.

Eigenschaften:

* Modellierung langfristiger zeitlicher Abhängigkeiten
* Erkennung komplexer saisonaler Muster
* Verarbeitung multivariater Zeitreihen

PatchTST verwendet sogenannte **Patches**, um lange Sequenzen effizienter durch Self-Attention-Mechanismen zu verarbeiten.

---

## Statistische Modelle

### Prophet & SARIMAX

Diese Modelle dienen als klassische Baselines.

Sie modellieren:

* Trends
* saisonale Komponenten
* wiederkehrende Muster

Ihre Vorteile liegen insbesondere in der Interpretierbarkeit und einfachen Modellierung einzelner Zeitreihen.

---

# 4. Evaluierungs-Framework

Um realistische Prognoseszenarien abzubilden, wurde eine strikt chronologische Validierungsstrategie ohne Data Leakage verwendet.

## Phase 1: Modellvalidierung

Notebook:

```
04_notebooks/04_model_training.ipynb
```

Training:

```
2013 - 2016
```

Validierung:

```
Januar 2017 - Mai 2017
```

Ziele:

* Vergleich der Modellarchitekturen
* Hyperparameter-Tuning
* Auswahl geeigneter Modellkonfigurationen

Beispielparameter:

```
LOOKBACK = 336
HORIZON = 76
```

---

## Phase 2: Finaler Test

Notebook:

```
04_notebooks/05_model_test.ipynb
```

Nach Auswahl der besten Konfigurationen werden die Modelle erneut trainiert.

Training:

```
2013 - Mai 2017
```

Finaler Test:

```
Juni 2017 - August 2017
```

Dieser Zeitraum bleibt vollständig isoliert und dient als realistische Simulation zukünftiger Prognosen.

---

# 5. Zentrale Ergebnisse & wissenschaftliche Erkenntnisse

Die Untersuchung zeigt, dass die Modellleistung stark von der Kombination aus geeigneten Features, Validierungsstrategie und Modellarchitektur abhängt.

Die wichtigsten Erkenntnisse:

* **Feature Engineering ist ein zentraler Erfolgsfaktor:**  
  Kalenderinformationen, Feiertage, Rolling Statistics und historische Benchmarks ermöglichen eine bessere Erfassung saisonaler und kurzfristiger Veränderungen.

* **Tree-basierte Modelle profitieren besonders von strukturierten Features:**  
  XGBoost und LightGBM können komplexe Zusammenhänge zwischen externen Faktoren und Verkaufszahlen effektiv nutzen.

* **Deep-Learning-Modelle erfassen langfristige Muster:**  
  PatchTST und NHITS eignen sich besonders für komplexe Sequenzen mit mehreren saisonalen Komponenten.

* **Statistische Modelle liefern wichtige Baselines:**  
  Prophet und SARIMAX ermöglichen eine transparente Modellierung von Trend und Saisonalität.

* **Zeitbasierte Evaluation verhindert unrealistische Ergebnisse:**  
  Durch die chronologische Aufteilung wird verhindert, dass Informationen aus zukünftigen Zeitpunkten in das Training gelangen.

Die entwickelte Pipeline ermöglicht dadurch einen reproduzierbaren Vergleich verschiedener Forecasting-Ansätze für komplexe Retail-Zeitreihen.

---

# ⚙️ Installation & Environment Setup

## Voraussetzungen

Empfohlen:

* Python >= 3.10
* Git
* virtuelle Python-Umgebung (`venv` oder `conda`)

---

## Repository klonen

```bash
git clone <repository-url>

cd Favorita-Store-Sales-Forecasting
```

---

## Virtuelle Umgebung erstellen

### Mit venv

Linux / macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## Abhängigkeiten installieren

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## Jupyter Kernel registrieren

Damit die Notebooks die virtuelle Umgebung verwenden:

```bash
python -m ipykernel install \
--user \
--name favorita_forecasting \
--display-name "Favorita Forecasting"
```

Anschließend im Notebook auswählen:

```
Kernel → Change Kernel → Favorita Forecasting
```

---

# ▶️ Ausführung der Pipeline

Die Pipeline wird schrittweise ausgeführt:

## 1. Allgemeiner Überblick

```
04_notebooks/01_data_check.ipynb
```
Enthält:

* Allgemeinen Überblick

## 2. Explorative Analyse


```
04_notebooks/02_eda.ipynb
```

Enthält:

* Datenverständnis
* Analyse von Trends und Saisonalitäten
* Untersuchung externer Einflussfaktoren

---

## 3. Preprocessing & Feature Engineering

```
04_notebooks/03_feature_engineering.ipynb
```

Enthält:

* Datenbereinigung
* Feature-Erstellung
* Aggregationen
* Export als Parquet-Dateien

---

## 4. Modelltraining

```
04_notebooks/04_model_training.ipynb
```

Enthält:

* Training aller Modellklassen
* Validierung
* Modellvergleich

---

## 5. Finaler Test

```
04_notebooks/05_model_test.ipynb
```

Enthält:

* Retraining mit erweiterten Trainingsdaten
* finale Evaluation
* Analyse der Prognosequalität

---

## 6. Hyptohesen Test

```
04_notebooks/06_hypothesen_analyse.ipynb
```

Enthält:

* Alle zugehörigen Analysen zu den Hyptohesen

---

# 📂 Repository-Struktur

```
├── 01_business_understanding/
│   └── Projektdefinition und Zielsetzung

├── 02_data/
│   ├── raw/
│   ├── processed/
│   └── final/

├── 03_src/
│   ├── Config
│   ├── Utility Functions
│   ├── Feature Engineering

├── 04_notebooks/
│   ├── 01_data_check.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_test.ipynb
│   └── 06_hypothesen_analyse.ipynb


├── requirements_clean.txt

└── README.md
```

---

# 📌 Reproduzierbarkeit

Alle Experimente können durch die dokumentierte Environment-Konfiguration und die chronologische Trainingsstrategie reproduziert werden.

Die Pipeline ist modular aufgebaut und kann einfach um weitere Modelle, Features oder externe Einflussfaktoren erweitert werden.