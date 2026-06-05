# Favorita Store Sales Forecasting: Multi-Model Pipeline

Dieses Projekt befasst sich mit der Vorhersage von Verkaufszahlen für die ecuadorianische Supermarktkette **Favorita**. Die zentrale Herausforderung liegt in der Verarbeitung von volatilen Zeitreihen, die stark von externen Faktoren wie Feiertagen, Ölpreisen und lokalen Events beeinflusst werden.

## 🛠 Methodik & Umsetzung

Anstatt auf ein einzelnes Modell zu setzen, wurde eine skalierbare **Vergleichs-Pipeline** entwickelt, die verschiedene mathematische Ansätze evaluiert:

### 1. Daten-Architektur & Performance
* **High-Speed Processing:** Konsequenter Einsatz von **Polars** für das Data Wrangling, um eine extrem speichereffiziente und multi-threaded Verarbeitung zu gewährleisten.
* **Storage:** Nutzung des **Parquet-Formats** zur optimierten, komprimierten Speicherung und schnellen Bereitstellung der Trainings- und Validierungsdaten.
* **Modularität:** Strikte Trennung der Kern-Logik (`03_src/`) von der interaktiven Analyse (`04_notebooks/`), um die Wiederverwendbarkeit und Wartbarkeit des Codes zu sichern.

### 2. Multivariate Feature Engineering
Die Zeitreihen wurden um kontextbezogene Dimensionen erweitert, um den Modellen tieferes Wissen über die Marktgegebenheiten zu vermitteln:
* **Ökonomische Indikatoren:** Integration des täglichen Ölpreises als Proxy für die gesamtwirtschaftliche Kaufkraft Ecuadors.
* **Kalendarische Ereignisse:** Mapping von nationalen und lokalen Feiertagen inklusive "Pre- & Post-Holiday"-Fenstern, um verändertes Kundenverhalten vor und nach freien Tagen einzufangen.
* **Zeitreihen-Dynamik:** Generierung von rollierenden Statistiken (Rolling Means über 7, 14, 28 Tage) und komplexen Zeit-Lags, um saisonale Trends (Wochentage, Monatsenden) explizit abzubilden.

### 3. Evaluierungs-Framework & Zweistufiger Workflow
Das Projekt folgt einem strikten, chronologischen Validierungsprozess ohne Daten-Leckage (*Data Leakage*), aufgeteilt in zwei aufeinander aufbauende Phasen:

* **Phase 1: Validierung (`04_model_training.ipynb`)** Modelle werden ausschließlich auf historischen Daten (2013–2016) trainiert und gegen ein isoliertes Validierungsfenster (Januar–Mai 2017) getestet. Diese Phase dient dem Benchmark der Modellarchitekturen und dem Tuning der Hyperparameter (z. B. `LOOKBACK = 336`, `HORIZON = 76`).
* **Phase 2: Finaler Test (`05_model_test.ipynb`)** Die Validierungsdaten werden in den Trainingsprozess integriert (`train_val`), um den Modellen die jüngste und wertvollste Historie bereitzustellen. Alle Modelle werden mit den fixierten Parametern re-trainiert und absolvieren auf dem blockierten Test-Zeitraum (Juni–August 2017) den finalen Härtetest.

Drei Modell-Generationen treten hierbei gegeneinander an:
* **Gradient Boosted Trees (XGBoost, LightGBM):** Globale, featurebasierte Modelle, optimiert auf einer 2D-Feature-Matrix.
* **Deep Learning (PatchTST, NHITS):** Sequenzbasierte neuronale Netze via `neuralforecast`, die komplexe multivariate Abhängigkeiten über Zeitfenster (*Patches*) hinweg lernen.
* **Statistische Modelle (Prophet, SARIMAX):** Lokale, pro Store isoliert trainierte Modelle, die als starke statistische Baseline dienen.

### 4. Zentrale Ergebnisse & wissenschaftliche Erkenntnisse


## 📂 Repository-Struktur

```text
├── 01_business_understanding/  # Projektdefinition und Zielsetzung
├── 02_data/                    # Strukturierte Ablage (Raw, Final, Results)
├── 03_src/                     # Kern-Logik (Features, Utilities, Config)
├── 04_notebooks/               # Workflow von EDA bis Evaluierung
│   ├── 01_eda.ipynb            # Explorative Datenanalyse & Saisonalitäten
│   ├── 02_preprocessing.ipynb  # Polars Feature Engineering & Parquet Export
│   ├── 04_model_training.ipynb # Phase 1: Training & Validierungs-Benchmark
│   └── 05_model_test.ipynb     # Phase 2: Re-Run auf Train+Val & Hypothesen-Analyse
├── requirements_clean.txt      # Projekt-Abhängigkeiten
└── README.md                   # Projektdokumentation