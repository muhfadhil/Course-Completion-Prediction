
# Course Completion Course

**Latar Belakang**

Seiring dengan meningkatnya popularitas pendidikan online, institusi pendidikan dan platform e-learning mencari cara untuk meningkatkan tingkat penyelesaian kursus. Banyak peserta kursus online mengalami kesulitan untuk menyelesaikan kursus yang mereka ikuti, yang dapat disebabkan oleh berbagai faktor seperti motivasi, waktu, dan kesulitan materi. Mengetahui faktor-faktor yang mempengaruhi penyelesaian kursus dan dapat memprediksi peserta mana yang kemungkinan besar tidak akan menyelesaikan kursus sangat penting untuk merancang intervensi yang efektif.

**Tujuan Proyek**

Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi apakah seorang peserta akan menyelesaikan kursus atau tidak berdasarkan berbagai fitur yang relevan. Dengan prediksi ini, platform e-learning dapat mengambil langkah proaktif untuk meningkatkan tingkat penyelesaian kursus, seperti memberikan dukungan tambahan kepada peserta yang berisiko.



## Dataset

Sumber: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-course-engagement-dataset

Dataset yang digunakan dalam proyek ini diambil dari Kaggle, dengan judul "Predict Online Course Engagement Dataset" yang dapat diakses di sini. Pada dataset tersebut terdapat 9000 baris dan 9 kolom, dengan kolom sebagai berikut:

- **UserID:** Identifikasi unik untuk setiap pengguna.
- **CourseCategory:** Kategori kursus yang diambil.
- **TimeSpentOnCourse:** Waktu yang dihabiskan peserta pada kursus dalam jam.
- **NumberOfVideosWatched:** Jumlah video yang ditonton oleh peserta.
- **NumberOfQuizzesTaken:** Jumlah kuis yang diambil oleh peserta.
- **QuizScores:** Rata-rata skor yang diperoleh peserta dari kuis.
- **CompletionRate:** Tingkat penyelesaian materi kursus.
- **DeviceType:** Jenis perangkat yang digunakan peserta untuk mengakses kursus (Dekstop (0) or Mobile (1)).
- **CourseCompletion:** Apakah peserta menyelesaikan kursus atau tidak (label target).
## Run Locally

Jalankan file `run.py`

```bash
  python run.py
```

Jalankan pada browser

```bash
  http://localhost:5000
```

## Documentation

Tampilan Course Completion Prediction App

![Course Completion Prediction App](https://github.com/muhfadhil/Course-Completion-Prediction/blob/main/docs/Course_Completion_Prediction_App.png?raw=true)

Demo hasil prediksi

![Demo App](https://github.com/muhfadhil/Course-Completion-Prediction/blob/main/docs/Demo_Result.png?raw=true)

## Authors

- [Muhammad Fadhil](https://www.github.com/muhfadhil)

