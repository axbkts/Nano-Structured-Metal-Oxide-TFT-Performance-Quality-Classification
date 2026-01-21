# Nano-Structured-Metal-Oxide-TFT-Performance-Quality-Classification
Quality/Performance classification of Nano-Structured Metal Oxide Thin-Film Transistors using 6 ML models (KNN, Logistic Reg, Naive Bayes, Decision Trees, SVM, Random Forest). / Nano-Yapılı Metal Oksit İnce Film Transistörlerin 6 ML modeli (KNN, Lojistik Reg, Naive Bayes, Karar Ağaçları, SVM, Rassal Orman) ile kalite sınıflandırması.

# Nano-Structured Metal Oxide TFT Performance & Quality Classification

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

**Key Features:**
* `Substrate_Type`: Flexible base material (Polyimide, PEN).
* `Mobility_cm2V-1s-1`: Charge carrier mobility indicating conductivity.
* `On_Off_Ratio`: Ratio of ON current to OFF current (Switching quality).
* `Bending_Radius_mm`: Minimum radius before performance degradation.
* `SNN_Accuracy_%`: Accuracy of the integrated spiking neural network.
* `Power_Consumption_mW`: Energy efficiency metric.

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
    python main.py
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

**Öne Çıkan Özellikler:**
* `Substrate_Type`: Esnek taban malzemesi (Polyimide, PEN).
* `Mobility_cm2V-1s-1`: Yük taşıyıcı mobilitesi (İletkenlik göstergesi).
* `On_Off_Ratio`: Anahtarlama kalitesi (Açık/Kapalı akım oranı).
* `Bending_Radius_mm`: Performans kaybı öncesi dayanılan minimum bükülme yarıçapı.
* `SNN_Accuracy_%`: Çip üzerindeki Spiking Neural Network (Yapay Sinir Ağı) başarımı.

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
    python main.py
    ```
