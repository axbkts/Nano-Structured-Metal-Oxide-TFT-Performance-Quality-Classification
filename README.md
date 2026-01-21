# Nano-Structured Metal Oxide TFT Performance & Quality Classification
Quality/Performance classification of Nano-Structured Metal Oxide Thin-Film Transistors using 6 ML models (KNN, Logistic Reg, Naive Bayes, Decision Trees, SVM, Random Forest). / Nano-Yapılı Metal Oksit İnce Film Transistörlerin 6 ML modeli (KNN, Lojistik Reg, Naive Bayes, Karar Ağaçları, SVM, Rassal Orman) ile kalite sınıflandırması.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

<p align="center">
  <b><a href="#-english">English</a></b> | 
  <b><a href="#-turkce">Türkçe</a></b>
</p>

---

<a name="-english"></a>
## English

### Project Overview
This project aims to classify the **Device Quality** (High, Medium, Low) of nano-structured metal-oxide thin-film transistors (TFTs) designed for next-generation flexible and conformable electronic devices. 

Using a specialized dataset containing 1,000 records of electrical, mechanical, and computational performance characteristics, we developed a comparative study using six different machine learning algorithms. The project investigates how critical material choices (e.g., IGZO vs. ZnO) and physical parameters (e.g., Bending Radius, Nanostructure Size) impact the final efficiency of the device.

### Dataset Description
The dataset represents the performance metrics of flexible TFT circuits.
* **Source:** Nano_Structured_Metal_Oxide_TFT_Dataset
* **Size:** 1000 Records, 21 Features
* **Target:** `Device_Quality`

#### Detailed Feature Descriptions
| Feature | Description |
| :--- | :--- |
| **Substrate_Type** | Type of flexible base material (e.g., Polyimide, PEN). |
| **Deposition_Method** | Fabrication approach used (ALD, Sputtering, Solution). |
| **Material_Type** | Composition of the oxide semiconductor (IGZO, ZnO, Hybrid). |
| **Film_Thickness_nm** | Physical thickness of the deposited metal-oxide layer. |
| **Annealing_Temp_C** | Temperature used for film stabilization during manufacturing. |
| **Nanostructure_Size_nm** | Size of structural nano-features in the film. |
| **Mobility_cm2V-1s-1** | Charge carrier mobility indicating conductivity (Higher is faster). |
| **Threshold_Voltage_V** | Gate voltage required to turn on the transistor. |
| **On_Off_Ratio** | Ratio of ON current to OFF current representing switching quality. |
| **Subthreshold_Swing_Vdec** | Efficiency of gate control over channel current. |
| **Gate_Density_gates_per_mm2** | Integration density of logic circuits per area. |
| **Bending_Radius_mm** | Minimum radius the device can bend before performance degradation. |
| **Cycles_to_Failure** | Durability measure under repeated bending cycles. |
| **Power_Consumption_mW** | Energy required for device operation. |
| **SNN_Accuracy_%** | Accuracy of the integrated Spiking Neural Network task. |
| **Response_Time_ms** | Average computational latency. |
| **Temperature_Stability_%** | Stability of performance across varying temperatures. |
| **Transparency_%** | Optical transparency of the film (for transparent electronics). |
| **Device_Quality** | **Target Variable:** Categorical label (High/Medium/Low). |

### Methodology & Preprocessing
To ensure robust model performance, the following pipeline was implemented:
1.  **Data Cleaning:** Checked for missing values and duplicates.
2.  **Feature Encoding:**
    * **One-Hot Encoding** for categorical features (`Substrate_Type`, `Deposition_Method`, `Material_Type`).
    * **Label Encoding** for the target variable (`Device_Quality`).
3.  **Scaling:** Applied **StandardScaler** to normalize numerical features, essential for distance-based algorithms like KNN and SVM.
4.  **Splitting:** Data was split into 80% Training and 20% Testing sets using **Stratified Sampling** to handle potential class imbalances.

### Models Implemented
We trained and evaluated the following classification algorithms:
1.  **K-Nearest Neighbors (KNN)** (k=5)
2.  **Logistic Regression**
3.  **Naive Bayes (Gaussian)**
4.  **Decision Trees**
5.  **Support Vector Machines (SVM)**
6.  **Random Forest Classifier**

### Results
The models were compared based on **Accuracy**, **F1-Score**, and **Confusion Matrices**.

*(Please upload the generated plots here after running the code)*

### Real-World Applications & Deployment
This classification model serves as a quality control gatekeeper in semiconductor manufacturing pipelines. By exporting the trained model (e.g., using `joblib` or `pickle`), it can be integrated into the following systems:

1.  **Automated Quality Assurance (QA):**
    * *Usage:* Integrate into the fabrication line's testing software.
    * *Benefit:* Instantly flag defective TFT batches before they are assembled into expensive flexible screens, reducing waste.
2.  **Predictive Maintenance for Wearables:**
    * *Usage:* Embed the model into the firmware of flexible health monitors.
    * *Benefit:* Monitor the degradation of transistor performance (via `Cycles_to_Failure` and `Mobility`) and alert the user before the device fails.
3.  **R&D Optimization:**
    * *Usage:* Use the model to simulate millions of material combinations (`Material_Type` vs `Nanostructure_Size`).
    * *Benefit:* Accelerate the discovery of high-performance materials without physical prototyping.

### How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/axbkts/Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification.git](https://github.com/axbkts/Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification.git)
    cd Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib
    ```

3.  **Run the analysis:**
    ```bash
    python nano_tft_classification.py
    ```

---

<a name="-turkce"></a>
## Türkçe

### Proje Özeti
Bu proje, esnek ve bükülebilir elektronik cihazlar için tasarlanmış **Nano-yapılı Metal-Oksit İnce Film Transistörlerin (TFT)** performans karakterlerini analiz ederek, cihaz kalitesini (**Device_Quality**) sınıflandırmayı amaçlamaktadır.

Elektriksel, mekanik ve hesaplama (SNN doğruluğu) özelliklerini içeren 1.000 kayıttan oluşan bir veri seti kullanılarak, 6 farklı makine öğrenmesi algoritması karşılaştırmalı olarak analiz edilmiştir. Çalışma, materyal seçiminin ve üretim yöntemlerinin cihazın "Yüksek", "Orta" veya "Düşük" kalite sınıfına girmesindeki etkisini ortaya koymaktadır.

### Veri Seti Hakkında
Veri seti, esnek TFT devrelerinin performans metriklerine odaklanmaktadır.
* **Veri Seti:** Nano_Structured_Metal_Oxide_TFT_Dataset
* **Boyut:** 1000 Satır, 21 Sütun
* **Hedef Değişken:** `Device_Quality` (High, Medium, Low)

#### Detaylı Özellik Açıklamaları
| Özellik (Sütun) | Açıklama |
| :--- | :--- |
| **Substrate_Type** | Esnek taban malzemesi tipi (Örn: Polyimide, PEN). |
| **Deposition_Method** | Üretim/Kaplama yöntemi (ALD, Sputtering, Solüsyon). |
| **Material_Type** | Oksit yarı iletken bileşimi (IGZO, ZnO, Hibrit). |
| **Film_Thickness_nm** | Depolanan metal-oksit katmanının kalınlığı (nm). |
| **Annealing_Temp_C** | Film stabilizasyonu için kullanılan tavlama sıcaklığı. |
| **Nanostructure_Size_nm** | Filmdeki nano yapıların boyutu. |
| **Mobility_cm2V-1s-1** | Yük taşıyıcı mobilitesi (İletkenlik hızı). |
| **Threshold_Voltage_V** | Transistörü açmak için gereken kapı (gate) gerilimi. |
| **On_Off_Ratio** | Açık/Kapalı akım oranı (Anahtarlama kalitesi). |
| **Subthreshold_Swing_Vdec** | Kapı kontrolünün kanal akımı üzerindeki verimliliği. |
| **Gate_Density_gates_per_mm2** | Mantık devrelerinin entegrasyon yoğunluğu. |
| **Bending_Radius_mm** | Performans bozulmadan önceki minimum bükülme yarıçapı. |
| **Cycles_to_Failure** | Bükülme döngülerine karşı dayanıklılık sayısı. |
| **Power_Consumption_mW** | Cihazın çalışması için gereken enerji. |
| **SNN_Accuracy_%** | Entegre Spiking Neural Network (Yapay Sinir Ağı) doğruluğu. |
| **Response_Time_ms** | Ortalama hesaplama/tepki süresi. |
| **Temperature_Stability_%** | Farklı sıcaklıklarda performansın kararlılığı. |
| **Transparency_%** | Filmin optik şeffaflık yüzdesi. |
| **Device_Quality** | **Hedef Değişken:** Cihaz Performans Sınıfı (Yüksek/Orta/Düşük). |

### Yöntem ve Ön İşleme
Modellerin başarısını artırmak için şu adımlar izlenmiştir:
1.  **Veri Temizleme:** Eksik veri ve aykırı değer analizi.
2.  **Kodlama (Encoding):**
    * Kategorik değişkenler için **One-Hot Encoding** (`Material_Type` vb.).
    * Hedef değişken için **Label Encoding**.
3.  **Ölçeklendirme (Scaling):** Tüm sayısal veriler **StandardScaler** ile normalize edilmiştir.
4.  **Veri Bölme:** Sınıf dengesini korumak için **Stratified (Tabakalı)** yöntemle %80 Eğitim, %20 Test ayrımı yapılmıştır.

### Kullanılan Modeller
Aşağıdaki algoritmalar proje kapsamında test edilmiştir:
1.  **K-En Yakın Komşu (KNN)** (k=5)
2.  **Lojistik Regresyon**
3.  **Naive Bayes (Gaussian)**
4.  **Karar Ağaçları (Decision Trees)**
5.  **Destek Vektör Makineleri (SVM)**
6.  **Rassal Orman (Random Forest)**

### Sonuçlar
Modeller **Doğruluk (Accuracy)**, **F1-Skoru** ve **Karmaşıklık Matrisleri (Confusion Matrix)** üzerinden değerlendirilmiştir.

*(Kodu çalıştırdıktan sonra oluşan grafiklerinizi buraya ekleyebilirsiniz)*

### Gerçek Dünya Uygulamaları ve Entegrasyon
Bu sınıflandırma modeli, yarı iletken üretim hatlarında bir "kalite kontrol bekçisi" olarak görev yapabilir. Eğitilen model dışarı aktarılarak (örn: `joblib` veya `pickle` ile) şu sistemlere entegre edilebilir:

1.  **Otomatik Kalite Kontrol (QA):**
    * *Kullanım:* Üretim bandındaki test yazılımlarına entegre edilir.
    * *Fayda:* Hatalı veya düşük kaliteli TFT partilerini, pahalı esnek ekranlara montajlanmadan önce tespit ederek israfı önler.
2.  **Giyilebilir Teknolojilerde Kestirimci Bakım:**
    * *Kullanım:* Esnek sağlık monitörlerinin gömülü yazılımına (firmware) entegre edilir.
    * *Fayda:* Transistör performansındaki düşüşü (`Cycles_to_Failure` ve `Mobility` verileriyle) izleyerek, cihaz bozulmadan önce kullanıcıyı uyarır.
3.  **Ar-Ge Optimizasyonu:**
    * *Kullanım:* Model, milyonlarca farklı materyal kombinasyonunu (`Material_Type` vs `Nanostructure_Size`) simüle etmek için kullanılır.
    * *Fayda:* Fiziksel prototip üretmeye gerek kalmadan yüksek performanslı materyallerin keşfini hızlandırır.

### Kurulum ve Çalıştırma
1.  **Projeyi indirin:**
    ```bash
    git clone [https://github.com/axbkts/Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification.git](https://github.com/axbkts/Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification.git)
    cd Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib
    ```

3.  **Kodu çalıştırın:**
    ```bash
    python nano_tft_classification.py
    ```
