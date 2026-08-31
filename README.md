# Ömer Faruk Akbay — Projects

**Electrical & Electronics Engineer, B.Sc.** (Bilkent University, 2026) — working where deep learning for imaging meets embedded hardware.

Computer vision and image super-resolution · embedded and edge systems · analogue and power electronics.

📍 Relocating to North Rhine-Westphalia, Germany · available from November 2026
🔗 [LinkedIn](#) · ✉️ faruk.akbay@ug.bilkent.edu.tr

---

## 🛰️ Featured — SR-SAR: Super Resolution for Synthetic Aperture Radar

**Senior design project with [Meteksan Savunma Sanayi](https://www.meteksan.com/), 2025–2026**
📄 [Project page](https://ee.bilkent.edu.tr/fuar/2026/group_c5/project_page_c5.html) · [Booklet (PDF)](https://ee.bilkent.edu.tr/fuar/2026/group_c5/c5_booklet.pdf)

A software-only deep-learning pipeline that improves the interpretability of MILSAR radar imagery at **2× and 4×** — with no change to the radar hardware.

| | |
|---|---|
| **Architecture** | Swin2-MoSE (Transformer-based super-resolution), PyTorch |
| **Core idea** | Dual-model strategy — one model optimised for **edge preservation**, one for **speckle suppression** |
| **Blending** | Outputs combined through an **α-controlled weighted blending** mechanism, so an operator can tune the structural-fidelity / noise-reduction trade-off |
| **Pipeline** | Patch-based preprocessing for large-format SAR scenes; training, validation and benchmarking workflow |
| **Delivery** | GUI for visualisation, comparison and export — usable by radar engineers with no ML background |

**Why it's hard:** in SAR imagery, speckle noise and genuine physical structure look alike. A super-resolution model that suppresses one will happily suppress the other, and a model that hallucinates detail is worse than useless for a defence application. The dual-model + α-blending design exists to make that trade-off explicit and controllable rather than baked into the weights.

Team of six · Academic supervisor: Dr. Vakur B. Ertürk · Industry mentor: Aksay Fatih Öncel

> 📁 Code and report: [`Data Science and AI/Graduation Project`](./Data%20Science%20and%20AI/Graduation%20Project)

---

## 📂 Repository structure

### 🤖 [Data Science and AI](./Data%20Science%20and%20AI)

**Computer vision & deep learning**

| # | Project | What it covers |
|---|---|---|
| 21 | [Computer Vision — EuroSAT Classification](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/21%20-%20Computer%20Vision%20-%20EuroSAT%20Classification) | Satellite land-use classification with deep learning |
| 08 | [Computer Vision with OpenCV](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/08%20-%20Computer%20Vision%20with%20OpenCV) | Image processing, face detection, feature matching |
| NN1 | [Two-Layer Neural Network from Scratch](./Data%20Science%20and%20AI/Neural%20Network%20Project/01%20-%20Two%20Layer%20Neural%20Network%20from%20Scratch) | Forward/backward pass implemented by hand (cs231n) |
| NN2 | [Fully Connected Networks and Dropout](./Data%20Science%20and%20AI/Neural%20Network%20Project/02%20-%20Fully%20Connected%20Networks%20and%20Dropout) | Deep FC networks, regularisation |
| NN3 | [Convolutional Neural Networks](./Data%20Science%20and%20AI/Neural%20Network%20Project/03%20-%20Convolutional%20Neural%20Networks) | CNNs in both PyTorch and TensorFlow |

**Machine learning & data science**

| # | Project | What it covers |
|---|---|---|
| 01 | [Data Analysis with Pandas](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/01%20-%20Data%20Analysis%20with%20Pandas) | Data wrangling and analysis |
| 02 | [Data Visualization](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/02%20-%20Data%20Visualization) | Interactive charts, racing bar plots, COVID-19 data |
| 03 | [Iris Classification and Visualization](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/03%20-%20Iris%20Classification%20and%20Visualization) | EDA with Matplotlib |
| 04 | [Regression — House Price Prediction](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/04%20-%20Regression%20Analysis%20-%20House%20Price%20Prediction) | Linear regression, King County housing |
| 05 | [Classification and Clustering](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/05%20-%20Classification%20and%20Clustering) | Supervised and unsupervised methods |
| 06 | [Titanic Survival Prediction](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/06%20-%20Titanic%20Survival%20Prediction) | Classical ML classifiers |
| 07 | [NLP — Spam Detection](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/07%20-%20NLP%20-%20Spam%20Detection%20and%20Text%20Analysis) | Text analytics and classification |
| 09 | [Deep Learning — ANN & Preprocessing](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/09%20-%20Deep%20Learning%20-%20ANN%20and%20Data%20Preprocessing) | Scaling, PCA, imputation |
| 10 | [Titanic with Deep Learning](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/10%20-%20Titanic%20Survival%20with%20Deep%20Learning) | Neural approach to the same problem |
| 11 | [Apache Spark and Big Data](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/11%20-%20Apache%20Spark%20and%20Big%20Data) | Distributed processing with PySpark |
| 12 | [Web Scraping](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/12%20-%20Web%20Scraping) | Automated data extraction |
| 13 | [AutoML and EDA](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/13%20-%20AutoML%20and%20EDA) | Automated model selection |
| 14 | [Geographic Information Systems](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/14%20-%20Geographic%20Information%20Systems) | Interactive maps, geospatial data |
| 15 | [Time Series Analysis](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/15%20-%20Time%20Series%20Analysis) | Forecasting on climate and sales data |
| 16 | [Recommender Systems](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/16%20-%20Recommender%20Systems) | KNN, popularity, matrix factorisation |
| 17 | [MLOps — Model Deployment](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/17%20-%20MLOps%20-%20Model%20Deployment%20with%20Flask%20and%20Streamlit) | End-to-end pipeline with Flask & Streamlit |
| 18 | [Loan Prediction Classification](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/18%20-%20Loan%20Prediction%20Classification) | Credit approval modelling |
| 19 | [DAAD Scholarship Web Scraper](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/19%20-%20DAAD%20Scholarship%20Web%20Scraper) | Practical scraping tool |
| 20 | [Datathon AI Competition](./Data%20Science%20and%20AI/Machine%20Learning%20Projects/20%20-%20Datathon%20AI%20Competition) | Competition qualification round |

### 🔌 [Embedded Systems](./Embedded%20Systems)

| Project | What it covers |
|---|---|
| [FFT Audio Spectrum Analyzer](./Embedded%20Systems/FFT%20Audio%20Spectrum%20Analyzer) | Real-time audio spectrum analysis on a microcontroller — sampling, FFT, live display |

### ⚡ [Electronics](./Electronics)

| Project | What it covers |
|---|---|
| [Wideband Amplifier with AGC](./Electronics/Wideband%20Amplifier) | 200 kHz – 2 MHz amplifier with automatic gain control. LTSpice simulation → DipTrace schematic & PCB → fabrication → bench validation. Stable 1 V peak output at < 100 mA. [Demo video](https://youtu.be/c0oU9KI2KH8) |

### 📜 [Certificates](./Certificates)

Professional certifications and course completions.

---

## 🧰 Tech

**Languages** Python · VHDL
**ML / AI** PyTorch · TensorFlow · Transformers · CNNs · OpenCV · scikit-learn · PySpark
**Embedded & edge** Rockchip RK3568 (DMA, RGA acceleration) · ARM Cortex-M0 · FFT/DSP on MCUs · Linux
**Electronics** KiCad · DipTrace · LTSpice · PCB layout · power electronics
**Digital design** VHDL on Digilent Basys 3 *(coursework)*
**Tools** Git · Jupyter · VS Code · Flask · Streamlit · AWS

---

## 📬 Contact

- **GitHub:** [@Omerfarukakbay](https://github.com/Omerfarukakbay)
- **Email:** faruk.akbay@ug.bilkent.edu.tr
