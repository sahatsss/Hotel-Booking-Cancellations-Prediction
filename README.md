# Hotel-Booking-Cancellations-Prediction

## Domain Proyek

Industri perhotelan merupakan salah satu sektor jasa yang sangat bergantung pada sistem reservasi. Salah satu tantangan utama yang dihadapi oleh hotel adalah tingginya tingkat pembatalan reservasi, yang dapat menyebabkan kerugian finansial akibat kamar yang dipesan namun tidak terisi. Berdasarkan studi [Antonio et al., 2017], tingkat pembatalan reservasi hotel dapat mencapai lebih dari 30% dari total booking, terutama yang dilakukan melalui agen perjalanan daring (OTA) dan reservasi dengan jangka waktu pemesanan yang panjang.

Masalah ini penting untuk diatasi karena memengaruhi beberapa aspek operasional hotel, seperti:

- Tingkat okupansi
- Perencanaan staf dan sumber daya
- Pendapatan yang hilang atau tidak terealisasikan

Dengan penerapan machine learning, kita dapat membangun sistem prediksi yang membantu hotel mengidentifikasi kemungkinan pembatalan sejak awal, sehingga dapat mengambil tindakan pencegahan seperti overbooking strategi, penyesuaian kebijakan pembayaran/deposit, atau kampanye follow-up ke customer.

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi apakah suatu reservasi akan dibatalkan atau tidak berdasarkan karakteristik booking tersebut?
2. Berdasarkan hasil prediksi apa yang harus diperbaiki oleh pihak hotel?

### Goals

1. Membangun model klasifikasi untuk memprediksi status reservasi (Canceled / Not Canceled)
2. Dengan model yang dibangun pihak hotel bisa menentukan langkah antisipasi jika pengunjung diprediksi membatalkan atau datang.
    ### Solution statements
    - Akan menggunakan 2 algoritma yang berbeda sebagai perbandingan, yaitu Random Forest dan Linear Regression

## Data Understanding

Dataset ini berisi data tentang pengunjung di hotel yang pernah melakukan booking di dua jenis hotel, yaitu resort dan kota. Ada fitur is_canceled dan reservation_status yang bisa digunakan sebagai label. Namun akan menggunakan is_canceled karena kurang distribusi pada kolom reservation_status.

- Jumlah data: 119,390 rows x 32 cols
- Kondisi Data:
Data kosong
country	488
agent	16340
company	112593
children 4

Duplikat: 0
[Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

 hotel:                          Jenis hotel city hotel atau resort hotel                                          
 is_canceled:                    Bisa menjadi label, tapi memilih menggunakan reservation_status. 1 jika reservasi dibatalkan, 0 jika tidak.                                                         
 lead_time:                      Jarak antara tanggal pemesanan dan tanggal kedatangan.
 arrival_date_year:              Tahun kedatangan tamu.                                                               
 arrival_date_month:             Bulan kedatangan tamu.                                                           
 arrival_date_week_number:       Minggu ke-berapa dalam setahun tamu datang.                                     
 arrival_date_day_of_month:      Tanggal kedatangan tamu dalam bulan tersebut.                                                                            
 stays_in_weekend_nights:        Jumlah malam menginap di akhir pekan.                                                                                    
 stays_in_week_nights:           Jumlah malam menginap di hari kerja.                                                                                     
 adults:                         Jumlah orang dewasa dalam reservasi.                                                                                     
 children:                       Jumlah anak-anak.                                                                                                        
 babies:                         Jumlah bayi.                                                                                                             
 meal:                           Jenis paket makan yang dipilih, seperti BB (bed & breakfast), HB (half board)                                    
 country:                        Negara asal                                              
 market_segment:                 Sumber pemesanan.
 distribution_channel:           Channel distribusi pemesanan.                                                               
 is_repeated_guest:              Apakah tamu adalah pelanggan yang pernah menginap sebelumnya (1) atau tidak (0).  
 previous_cancellations:         Jumlah pemesanan sebelumnya yang dibatalkan oleh tamu ini.                          
 previous_bookings_not_canceled: Jumlah pemesanan sebelumnya yang tidak dibatalkan.                                                                       
 reserved_room_type:             Tipe kamar yang diminta saat reservasi.                                                                                  
 assigned_room_type:             Tipe kamar yang sebenarnya diberikan.                           
 booking_changes:                Berapa kali pemesanan ini mengalami perubahan.                                                                           
 deposit_type:                   Jenis deposit yang diterapkan (no deposit, refundable, atau non refundable).                                              
 agent:                          ID agen perjalanan yang membuat reservasi. 0 berarti tanpa agen.                                                         
 company:                        ID perusahaan yang membuat reservasi, jika pemesanan dilakukan atas nama perusahaan.              
 days_in_waiting_list:           Jumlah hari reservasi berada dalam daftar tunggu.                                                                        
 customer_type:                  Jenis pelanggan: contract, group, transient, transient party.                                                            
 adr:                            rata-rata harga kamar per malam.                                                                
 required_car_parking_spaces:    Jumlah tempat parkir mobil yang diminta.                                                                                 
 total_of_special_requests:      Jumlah permintaan khusus dari tamu                                
 reservation_status:             Status akhir reservasi: canceled, check out, atau no show.                                                               
 reservation_status_date:        Tanggal ketika status terupdate                               
                
## Data Preparation

- Mengubah data types menjadi datetime pada kolom tanggal
- Menghapus fitur yang kira kira tidak berpengaruh seperti 'reservation_status_date','market_segment', 'distribution_channel','arrival_date_year','reservation_status', 'agent', 'company
- Menghapus missing value
- Mengubah seluruh tipe data float dengan angka di belakang koma menjadi int
- Menghapus data yang kira kira memiliki duplikat
- Melakukan encoding terhadap fitur kategorikal seperti meal, reservation_status, dll
- Melakukan splitting data menjadi X untuk fitur dan y untuk label
- Membagi X dan y menjadi train dan test dataset dengan label is_canceled

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
Pada project kali ini saya menggunakan 2 model yang berbeda yaitu Random Forest dan Logistic Regression. Setelah melakukan splitting menjadi data train dan test, model lalu di train. Namun setelah melakukan evaluasi, akurasi model terbilang cukup rendah, dimana Random Forest memiliki akurasi sekitar 79%, dan Logistic regression sebesar 73%. Kemudian melakukan tuning dengan metode Random Search.

### Parameter yang digunakan setelah tuning

- Random Forest
n_estimators=1000
Jumlah pohon dalam hutan. Semakin banyak pohon dapat meningkatkan akurasi, namun juga menambah waktu komputasi. Nilai 1000 dipilih untuk memberikan stabilitas hasil model.

min_samples_split=6
Jumlah minimum sampel yang dibutuhkan untuk membagi sebuah node internal.

min_samples_leaf=2
Jumlah minimum sampel yang dibutuhkan untuk menjadi daun pohon.

max_features='sqrt'
Menentukan jumlah fitur yang dipertimbangkan saat mencari split terbaik.

max_depth=30
Membatasi kedalaman maksimal pohon.

- Logistic Regression
'C': 10000.0
Parameter untuk regularisasi, semakin kecil berarti regularisasi kuat guna menghindari overfitting. Semakin besar artinya regularisasi lemah

penalty=['l1']
Jenis regularisasi. l1 dapat menghasilkan model sparse (fitur tak relevan = 0), sedangkan l2 lebih stabil dan umum digunakan.

solver=['liblinear']
Algoritma optimisasi yang mendukung regularisasi l1 dan l2. liblinear cocok untuk dataset kecil hingga menengah, sedangkan saga lebih baik untuk dataset besar dan mendukung l1.

max_iter=[500]
Jumlah iterasi maksimum selama proses optimasi.

### Modeling Comparison

- Random Forest
Random forest adalah ensemble learning method yang membentuk banyak decision tree dan menggabungkan prediksinya. Cocok untuk data yang non-linier dan kompleks.

- Logistic Regression
Logistic regression adalah model statistik linier yang digunakan untuk memprediksi probabilitas kelas dari suatu input. Cocok untuk data yang bersifat linier

## Evaluation

Accuracy (Akurasi)
Persentase prediksi yang benar dibandingkan total dataCocok digunakan jika dataset seimbang antara kelas positif dan negatif.

Precision (Presisi)
Seberapa akurat model saat memprediksi kelas positif.
Seamakin tinggi berarti semakin sedikit false positive 

Recall (Sensitivitas/True Positive Rate)
Seberapa baik model memahami semua kasus kelas positif.
Semakin tinggi maka semakin sedikit false negative 

F1 Score
Rata-rata gabungan dari precision dan recall. Cocok digunakan jika kita ingin keseimbangan antara keduanya.

- Random Forest
precision    recall  f1-score   support

           0       0.82      0.93      0.87     12371
           1       0.72      0.47      0.57      4680

    accuracy                           0.81     17051
   macro avg       0.77      0.70      0.72     17051
weighted avg       0.80      0.81      0.79     17051

- Logistic Regression
     precision    recall  f1-score   support

           0       0.77      0.95      0.85     12371
           1       0.65      0.25      0.36      4680

    accuracy                           0.76     17051
   macro avg       0.71      0.60      0.61     17051
weighted avg       0.74      0.76      0.72     17051

Random forest menunjukkan kinerja yang lebih seimbang dan kuat terutama dalam mendeteksi reservasi yang tidak dibatalkan (kelas mayoritas). Namun recall untuk kelas canceled (kelas 1) masih relatif rendah, meski precision-nya cukup baik. Untuk logistic regression memiliki recall yang sangat rendah pada kelas canceled (kelas 1), yang berarti model ini kurang mampu mengenali pelanggan yang akan membatalkan reservasi. Meskipun kelas 0 dikenali dengan sangat baik (recall 0.95), ketimpangan performa antar kelas mengindikasikan model ini tidak ideal untuk kasus dengan fokus pada prediksi pembatalan. Secara keseluruhan, Random forest memberikan performa yang lebih baik dibanding logistic regression dalam hal akurasi, keseimbangan antar kelas, dan kemampuan mengenali pembatalan (meski masih dapat ditingkatkan). Oleh karena itu, Random forest dipilih sebagai model terbaik untuk skenario klasifikasi status reservasi ini.

Dengan membangun model klasifikasi ini, kita sudah bisa menjawab pertanyaan pertama yaitu bagaimana cara mempridiksi manakah pelanggan yang kira kira memiliki kemungkinan untuk membatalkan pesanannya. Setelah melewati proses analisis data, bisa dilihat ada tren yang terlihat antara kenaikan harga dan tingkat pembatalan. Mungkin pihak hotel bisa menyesuaikan harga di periode tertentu untuk bisa mengurangi banyaknya pelanggan yang membatalkan pesanan pada periode tersebut. Dengan model prediksi dan analisis data diharapkan dapat berdampak pada performa hotel.
**---Ini adalah bagian akhir laporan---**
