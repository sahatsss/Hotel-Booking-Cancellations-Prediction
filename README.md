# Hotel Booking Cancellation Prediction

Model klasifikasi untuk memprediksi pembatalan reservasi hotel berdasarkan data Hotel Booking Demand.

## Ringkasan

Pembatalan reservasi menyebabkan kamar kosong dan kerugian pendapatan bagi hotel. Proyek ini membandingkan Random Forest dan Logistic Regression untuk memprediksi label `is_canceled`. Hyperparameter kedua model dicari dengan randomized search.

## Hasil

Evaluasi pada data uji (17.051 baris):

| Model | Accuracy | Macro F1 | Recall (canceled) |
|---|---|---|---|
| Random Forest | 0.81 | 0.72 | 0.47 |
| Logistic Regression | 0.76 | 0.61 | 0.25 |

Random Forest dipilih sebagai model akhir. Recall untuk kelas *canceled* masih rendah, sehingga penanganan ketidakseimbangan kelas menjadi langkah lanjutan yang paling relevan.

## Dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Kaggle). Dataset ini berisi data reservasi dari sebuah city hotel dan sebuah resort hotel.

## Struktur

```
.
├── hotel_cancellation_prediction.ipynb
├── hotel_bookings.csv
└── docs/
    └── laporan.md
```

## Menjalankan

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
jupyter notebook hotel_cancellation_prediction.ipynb
```

Sebelum dijalankan, ubah path pada `read_csv()` di notebook menjadi `hotel_bookings.csv`.

## Dokumentasi

Laporan lengkap (business understanding, data preparation, tuning, dan evaluasi) ada di [docs/laporan.md](docs/laporan.md).
