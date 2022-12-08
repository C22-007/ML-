# Machine Learning - C22-007

## Domain Proyek
Project Machine Learning : membuat model Predictive Analysis, menggunakan dataset yang berdomain kesehatan mengenai bodyfat.
### Latar Belakang
Pandemi memicu meningkatnya tingkat stres pada masyarakat. Stres memiliki keterkaitan dengan peningkatan kortisol. Peningkatan kortisol dipengaruhi oleh asupan makanan tinggi garam, lemak, atau keduanya. Stres dan kadar kortisol yang tinggi dikaitkan dengan peningkatan body fat yang  menyebabkan seseorang terkena resiko penyakit jantung dan diabetes tipe 2.
Menurut Mayo Clinic, persentase body fat yang sehat memiliki efek signifikan pada kadar kolesterol HDL yang membantu menghilangkan LDL yang merusak dan menurunkan kolesterol total secara keseluruhan. Untuk itu dibutuhkan pengukuran body fat dan solusi penyelesaian yang ideal dengan mengukur berapa persen body fat lalu dilanjutkan dengan adanya rekomendasi makanan berdasarkan informasi/data dari paper/jurnal di web.


Referensi: [Gender-based approach to estimate the human body fat percentage using Machine Learning](https://ieeexplore.ieee.org/abstract/document/9533512)

## Business Understanding
### Problem Statements
- Variable apa saja yang berpengaruh pada bodyfat seseorang ?
- Apakah variable yang ada pada dataset dapat tepat memprediksi bodyfat secara tepat ?
 
### Goals
- Mengetahui variable apa saja yang dapat mempengaruhi seseorang memiliki bodyfat berlebih
- Membuat model mechine learning sebaik mungkin dengan score yang tinggi berdasarkan variable variable yang ada 

### Solution Statements
Untuk mendapatkan model mechine learning pada prediksi diabetes yang saya buat. saya menggunakan algoritma Bayesian Ridge Regression
- BayesianRidge Regression adalah jenis pemodelan bersyarat di mana rata-rata satu variabel dijelaskan oleh kombinasi linier dari variabel lain, dengan tujuan untuk mendapatkan probabilitas posterior dari koefisien regresi serta parameter lain yang menggambarkan distribusi regresi dan akhirnya memungkinkan prediksi out-of-sample dari regresi dan tergantung pada nilai-nilai yang diamati dari para regresi.

## Data Understanding
Dataset yang saya gunakan merupakan dataset survey 253.680 orang responden yang sudah bersih. Dataset ini terdiri dari 22 kolom (variabel) yang semuanya bertipe data float64.


Variabel - variabel yang terdapat di Dataset :
- Density = berat dalam air
- BodyFat = persentase bodyfat 
- Age = umur
- Weight = berat(kg)
- Height = tinggi(cm)
- Neck = lingkar leher (cm)
- Chest = lingkar dada (cm)
- Abdomen = lingkar perut (cm)
- Hip = Lingkar Pinggul (cm)
- Thigh = Lingkar paha (cm)
- Knee = Lingkar lutut (cm)
- Ankle = Lingkar pergelangan kaki (cm)
- Biceps = Bisep (diperpanjang) lingkar (cm)
- Forearm = Lingkar lengan bawah (cm)
- Wrist = Lingkar pergelangan tangan (cm)
- BMI = berat badan / (tinggi badan)^2


Data loading 

|  Density |   BodyFat |  Age |  Weight |  Height | Neck | Chest | Abdomen |   Hip | Thigh | Knee | Ankle | Biceps | Forearm | Wrist |       BMI |
|---------:|----------:|-----:|--------:|--------:|-----:|------:|--------:|------:|------:|-----:|------:|-------:|--------:|------:|----------:|
| 1.059463 | 17.217614 | 23.0 | 69.4125 | 172.085 | 36.2 |  93.1 |    85.2 |  94.5 |  59.0 | 37.3 |  21.9 |   32.0 |    27.4 |  17.1 | 23.439679 |
| 1.060776 | 16.639464 | 22.0 | 77.9625 | 183.515 | 38.5 |  93.6 |    83.0 |  98.7 |  58.7 | 37.3 |  23.4 |   30.5 |    28.9 |  18.2 | 23.149554 |
| 1.057177 | 18.228062 | 22.0 | 69.3000 | 168.275 | 34.0 |  95.8 |    87.9 |  99.2 |  59.6 | 38.9 |  24.0 |   28.8 |    25.2 |  16.6 | 24.473385 |
| 1.054530 | 19.403411 | 26.0 | 83.1375 | 183.515 | 37.4 | 101.8 |    86.4 | 101.2 |  60.1 | 37.3 |  22.8 |   32.4 |    29.4 |  18.2 | 24.686176 |
| 1.053868 | 19.698343 | 24.0 | 82.9125 | 180.975 | 34.4 |  97.3 |   100.0 | 101.9 |  63.2 | 42.2 |  24.0 |   32.2 |    27.7 |  17.7 | 25.315286 |
|      ... |       ... |  ... |     ... |     ... |  ... |   ... |     ... |   ... |   ... |  ... |   ... |    ... |     ... |   ... |       ... |
| 1.042255 | 24.931739 | 70.0 | 60.4125 | 170.180 | 34.9 | 89.2  | 83.6    | 88.8  | 49.6  | 34.8 | 21.5  | 25.6   | 25.7    | 18.5  | 20.859782 |
| 1.020743 | 34.940714 | 72.0 | 90.4500 | 177.165 | 40.9 | 108.5 | 105.0   | 104.5 | 59.6  | 40.8 | 23.2  | 35.2   | 28.6    | 20.1  | 28.817262 |
| 1.018008 | 36.243853 | 72.0 | 84.0375 | 167.640 | 38.9 | 111.1 | 111.5   | 101.7 | 60.3  | 37.3 | 21.5  | 31.3   | 27.2    | 18.0  | 29.903211 |
| 1.025943 | 32.482744 | 72.0 | 85.8375 | 179.070 | 38.9 | 108.3 | 101.3   | 97.8  | 56.0  | 41.6 | 22.7  | 30.5   | 29.4    | 19.8  | 26.768953 |
| 1.017965 | 36.264459 | 74.0 | 93.3750 | 177.800 | 40.8 | 112.4 | 108.5   | 107.1 | 59.3  | 42.2 | 24.6  | 33.7   | 30.0    | 20.6  | 29.537049 |

deskripsi dataset

|           |    Density |    BodyFat |        Age |     Weight |     Height |       Neck |      Chest |    Abdomen |        Hip |      Thigh |       Knee |      Ankle |     Biceps |    Forearm |      Wrist |        BMI |
|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
|     count | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 | 252.000000 |
|      mean |   1.043832 |  24.269847 |  44.884921 |  80.305446 | 178.508075 |  37.967808 | 100.742163 |  92.428770 |  99.735268 |  59.328175 |  38.562500 |  23.038095 |  32.255605 |  28.675595 |  18.222222 | 25.121930  |
|       std |   0.011321 |   5.160921 |  12.602040 |  12.284289 |   6.751411 |   2.301730 |   8.161876 |  10.293612 |   6.438057 |   4.962811 |   2.321649 |   1.403545 |   2.958537 |   1.962956 |   0.911143 | 3.274559   |
|       min |   1.012335 |  12.440652 |  22.000000 |  53.325000 | 158.115000 |  31.862500 |  79.300000 |  69.400000 |  85.000000 |  47.200000 |  33.000000 |  19.100000 |  24.800000 |  23.250000 |  15.800000 | 17.874982  |
| 25.750000 |   1.037574 |  20.893605 |  35.750000 |  71.550000 | 173.355000 |  36.400000 |  94.350000 |  84.575000 |  95.500000 |  56.000000 |  36.975000 |  22.000000 |  30.200000 |  27.300000 |  17.600000 | 22.853241  |
| 50.500000 |   1.044006 |  24.135207 |  43.000000 |  79.425000 | 177.800000 |  38.000000 |  99.650000 |  90.950000 |  99.300000 |  59.000000 |  38.500000 |  22.800000 |  32.050000 |  28.700000 |  18.300000 | 24.895566  |
| 75.250000 | 1.051193   | 27.074634  | 54.000000  | 88.650000  | 183.515000 | 39.425000  | 105.375000 | 99.325000  | 103.525000 | 62.350000  | 39.925000  | 24.000000  | 34.325000  | 30.000000  | 18.800000  | 27.118085  |
| max       | 1.070408   | 38.968420  | 81.000000  | 114.300000 | 197.485000 | 43.962500  | 121.912500 | 121.450000 | 115.562500 | 71.875000  | 44.350000  | 27.000000  | 40.512500  | 34.050000  | 20.600000  | 33.515350  |

visualisasin data

histogram chart 

![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/download%20(2).png?raw=True)

relasi setiap variable 

![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/download%20(1).png?raw=True)

distribusi pada variable bodyfat

![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/download.png?raw=True)

melihat pairplot pada setiap variable dengan variable bodyfat

![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/pairplot%201.png?raw=True)
![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/pairplot%202.png?raw=True)
![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/pairplot%203.png?raw=True)



selain itu juga kita mengecek untuk apakah ada data outlier 
![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/boxplot%201.png?raw=True)

![MSE](https://github.com/alpiansyah1204/ml-terapan-s1/blob/main/image/outlier.jpeg?raw=True)

bisa dilihat dari gambar di atas data yang dipakai harus ada diantara -1.5(Q3-Q1) dan 1.5(Q3-Q1) atau bisa ditulis 
-1.5(Q3-Q1)<data<1.5(Q3-Q1)

setelah data outlier dibersihkan 
![MSE](https://github.com/C22-007/ML-/blob/main/image%20capstone/boxplot%202.png?raw=True)


## Data Preparation
Sebelum datasetnya di latih atau training, dari model sebelumnya perlu melakukan encoding lalu pemisahan data antara data latih dan test agar data dapat dilatih.


#### Train-Test Split
Proses splitting data atau pembagian dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum melakukan pemodelan supervised. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan train test split karena untuk efisiensi dan tidak melakukan data leakage ketika melakukan scaling. pada proyek kali ini kita membagi data menjadi 80:20 dengan random state = 93 

## Modeling

pada proyek yang dibuat kali ini, digunakan model algoritma mechine learning yaitu Machine Learning yaitu bayessianridge model tersebut dipilih karena tujuanya ingin memprediksi regresi. 

- pada BayesianRidge kita hanya menggunakan fungsi fit tanpa tambahan parameterlain 
`bay = BayesianRidge().fit(X_train, y_train)`
  - kelebihan pada algoritma ini yaitu mendapatkan pendekatan yang sepenuhnya matematis di mana Anda dapat dengan mudah memasukkan pengetahuan sebelumnya tentang domain masalah. Ini berfungsi dengan baik saat Anda memiliki sedikit data dan akan memberi Anda lebih dari sekadar prediksi; itu akan memberi Anda distribusi probabilitas penuh atas semua kemungkinan model linier, yang memungkinkan Anda untuk melakukan hal-hal seperti interval kepercayaan, penghindaran risiko, ... Selain itu, ini sangat cocok untuk pendekatan online karena Anda selalu dapat memasukkan lebih banyak data tanpa perlu menyimpan semua data sebelumnya; yang Anda butuhkan hanyalah model posterior saat ini.
  - kekurangan pada bayessionridge Menemukan prior yang baik bisa jadi sulit (walaupun saya menemukan prior informasi minimal bekerja dengan baik dalam praktiknya) dan ketika Anda memiliki banyak data dan hanya peduli untuk mempelajari satu model alih-alih distribusi model, Anda akan berakhir dengan pada dasarnya model yang sama dengan jenis regresi linier lainnya (dengan jumlah data yang cukup besar, pendekatan frequentist dan Bayesian cenderung menyatu dengan model yang sama).
  
  
## Evaluation
- R-Squared (coefficient of determination).
 -Disini saya menggunakan Metric Evaluation yaitu R^2_score atau R-squared. R-Squared itu sendiri adalah skor terbaik yang mungkin adalah 1,0 dan bisa negatif (karena modelnya bisa sewenang-wenang lebih buruk). Sebuah model konstan yang selalu memprediksi nilai yang diharapkan dari y, mengabaikan fitur input, akan mendapatkan skor 0,0.
  - untuk persamaannya seperti ini
 
    - ![R2-SQUARED MACHINE LEARNING](https://user-images.githubusercontent.com/64582353/135482517-1f589eb6-d59f-4872-8d9d-eddd673c1124.png)
- **Kelebihannya**
  - dapat memprediksi hasil di masa depan atau pengujian hipotesis , berdasarkan informasi terkait lainnya.
  - memberikan ukuran seberapa baik hasil yang diamati direplikasi oleh model, berdasarkan proporsi variasi total hasil yang dijelaskan oleh model.
  - sangat cocok untuk metrics akurasi pada model Regresi.
- **Kekurangan**
  - tidak menunjukan apakah regresi yang benar digunakan
  - tidak dapat memberitahu apakah model tersebut overfit/underfit dan lainnya.
  
  
 nilai yang didapat dari algortima Bayesian linear regression? yaitu 0.9966827666254439
