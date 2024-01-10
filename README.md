# Machine-Learning
Proyek Machine Learning Dicoding
# Laporan-ML-Terapan-1-Loisa M. T Allo
## Domain Proyek
### Mesin Kasir Mandiri 
Pengunaan mesin kasir pada store atau tempat berbelanja dimana ketika selesai berbelanja maka pembeli akan  melakukan *scanning* barang secara mandiri dan langsung melakukan pembayaran jadi tidak diperlukan lagi petugas di kasir. Mesin ini sudah banyak digunakan di australia, eropa dan negara maju lainnya.
Mesin ini dapat mempermudah dalam proses pelayanan dimana kita tidak memerlukan lagi pekerja tambahan di kasir pembayaran serta dapat menghemat waktu. Tentu saja mesin ini memiliki proses *Machine Learning* didalamnya jadi dapat dibuat sebuah model sederhana yang menyerupai cara kerja mesin ini. 
  
## Bussiness Understanding
### Problem Statement :
  - Bagaimana sistem ini dapat membantu *owner* dari suatu tempat perbelanjaan yang harus mencari lagi pekerja di bagian kasir untuk melakukan tugas *scanning* barang serta proses pembayaran tentu hal ini akan lebih memakan banyak waktu, tenaga dan juga biaya
  - Bagaimana menyelesaikan masalah proses pembayaran dan *scaning* dikasir yang biasanya memakan waktu yang lama dan antriannya panjang
### Goals :
  - *Owner* tidak perlu mencari lebih banyak pekerja untuk bagian kasir dengan sistem ini karena menggunakan machine learning yang bekerja secara mandiri
  - Membuat suatu sistem dimana pembeli dapat melakukan pembayaran dan proses *scanning* secara mandiri
#### Solution Statement :
  - Menggunakan image classification dan juga NLP
  - Image classification dapat digunakan untuk mengelompokkan barang sesuai dengan jenisnya seperti sayuran dan buah buahan
  - Untuk memperoleh sitem yang optimal tentu saja membutuhkan banyak dataset
  
## Data Understanding
### Dataset yang digunakan :
Data yang digunakan adalah data gambar atau image dari buah buahan untuk dibuat dalam image classification. Dataset ini dalam bentuk zip file dan memiliki banyak sampel buah yang sudah dikelompokkan sesuai namanya.
Data tersebut dapat diunduh disini : [*Dataset Model*](https://www.kaggle.com/datasets/sshikamaru/fruit-recognition). Data yang ada tidak digunakan semuanya, tetapi hanya sebagian karena terdapat banyak sekali jenis buah maka hanya diambil beberapa sampel buah untuk digunakan dalam model yang saya buat.
### Variabel-Variabel :
  - Avocado : Merupakan variabel yang berisi data gambar buah alpukat atau dalam bahasa inggris disebut avocado
  - Banana  : Merupakan variabel yang berisi data gambar buah pisang atau dalam bahasa inggris disebut banana
  - Blubbery : Merupakan variabel yang berisi data gambar buah bluberry
#### Tahapan penting :
Salah satu hal yang penting untuk diperhatikan adalah menganalisa data gambar buah apa saja yang ada didalam dataset tersebut dan background apa yang digunakan karena pada proyek sebelumnya background juga sangat berpengaruh dalam proses ini.

## Data Preparation
Proses ImageDataGenerator dalam data preparation :
Keras menyediakan ImageDataGenerator yang memudahkan untuk melakukan augmentasi gambar pada dataset yang digunakan.
  - Rescale : berfungsi untuk menormalisasi setiap nilai piksel pada gambar menjadi nilai antara 0 sampai 1. Nilai 1/255 artinya setiap nilai akan dikali 1/255 sehingga nilainya akan berubah menjadi antara 0 dan 1.
  - Rotation_range : berfungsi melakukan rotasi pada gambar secara acak
  - Shear_range : berfungsi untuk mengaplikasikan pergeseran sudut sebesar 0.2 = 20%
  - Zoom_range : berfungsi melakukan augmentasi berupa zoom pada gambar (0.2 = 20%)
  - Vertical_flip - Berkebalikan dengan horizontal flip, sesuai namanya vertical flip membalikkan gambar secara vertikal. diset nilainya sebagai True
  - Width_shift_range, angka yang digunakan 0.2. Angka ini menentukan seberapa besar gambar digeser secara acak, baik ke kanan ataupun ke kiri.
  - Height_shift_range : angka yang dugunakan 0.2. Perbedaannya dengan width terletak pada arah pergeseran, height_shift_range digeser secara vertical
  - Fill_mode berfungsi untuk mengisi wilayah atau gambar yang tidak memiliki nilai. Fill mode yang digunakan adalah Nearest: nilai piksel terdekat dari wilayah yang tidak memiliki nilai dipilih dan diulang.
  - Class_mode dibuat ‘categorical’ karena akan menentukan 3 class yaitu, Avoccado, Banana, dan Blubbery. Jika hanya membutuhkan 2 class, maka lebih baik menggunakan ‘binary’.
  - Target_size=(150, 150) : mengubah resolusi seluruh gambar menjadi 150x150 piksel
  
Beberapa hal yang saya lakukan pada tahap ini yaitu :
  - Data diseleksi terlebih dahulu karena banyak sekali data, maka hanya diambil beberapa sampel data untuk digunakan dalam model
  - Menggabungkan data yang sudah diseleksi kedalam satu file dengan nama 'Revisi Dataset'
  - Karena dataset yang dibaca dalam bentuk zip atau csv maka perlu untuk mengubah dataset yang sudah dibuat kedalam bentuk Zip File
  - Dalam Model yang dibuat data akan dibaca terlebih dahulu
  - Setelah data dibaca maka dibagi data tersebut kedalam traning dan test dan hasilnya ditampilkan terdapat 1379 gambar yang dibagi menjadi 3 kelas untuk training dan 748 gambar yang dibagi menjadi 3 kelas untuk test. 
  - Proses selanjutnya yaitu membuat model sequens, model compile, dan model fit dimana pada model fit ini saya akan melihat akurasi dari data yang telah dilatih
  - Proses terakhir dimana kita akan mengupload gambar atau image dan model akan mencoba memprediksi gambar yang telah kita upload
 
 #### Tahapan diatas perlu dilakukan karena :
  - Dataset yang telah didownload tidak semua datanya dipakai jadi perlu membuat dataset baru dari dataset yang telah didownload
  - Proses selanjutnya adalah tahapan-tahapan yang perlu diikuti agar model berhasil memprediksi gambar
  
 ## Modeling
 Dengan menggunakan Keras Sequential, layer-layer arsitektur jaringan saraf tiruan didefinisikan secara urut dari depan (input layer) sampai belakang (output layer). Model sequential ini memiliki tumpukan layer-layer, yang sama seperti pada sebuah MLP.
### Layer yang digunakan :
  - Conv2D : Proses konvolusi adalah proses yang mengaplikasikan filter pada gambar. Pada proses konvolusi ada perkalian matriks terhadap filter dan area pada gambar.
  Dengan ukuran (3, 3) dengan jumlah filter sebanyak 32 filter. Sehingga, tiap satu input gambar akan menghasilkan 16 gambar baru dengan ukuran (150, 150). 
  - MaxPooling2D : umumnya setelah proses konvolusi pada gambar masukan, akan dilakukan proses pooling. Pooling adalah proses untuk mengurangi resolusi gambar dengan tetap mempertahankan informasi pada gambar. resolusi tiap gambar akan diperkecil dengan tetap mempertahankan informasi pada gambar menggunakan MaxPoling layer yang berukuran (2, 2)
  - Flatten Layer : Output dari MaxPoling layer terakhir akan diubah ke dalam bentuk array 1D (tensor 1D). Fungsi flatten layer sendiri untuk membuat input yang memiliki banyak dimensi menjadi satu dimensi. Digunakan sebelum ke fully connected
  - Dropout Layer : Dropout mengacu pada unit/perseptron yang di-dropout (dibuang) secara temporer pada sebuah layer. Contohnya seperti pada model yang dibuat di mana besaran dropout yang dipilih adalah 0.8 sehingga 80% dari persepteron layer dimatikan secara berkala pada saat pelatihan.
  - Dense Layer : dense layer pertama yang memiliki 512 neuron. Sehingga, ia akan menghasilkan output dengan ukuran (512). Selanjutnya, output ini akan masuk pada dense layer kedua yang memiliki 3 neuron sehingga akan menghasilkan output dengan ukuran (3). Output dari layer terakhir inilah yang digunakan sebagai hasil akhir model untuk kasus klasifikasi categorial
  - activation = 'relu' : merupakan fungsi aktivasi pada hidden layer
  - activation = 'softmax' : Untuk dataset yang memiliki 3 kelas atau lebih, digunakan fungsi aktivasi Softmax. Fungsi aktivasi softmax akan memilih kelas mana yang memiliki probabilitas tertinggi
  
 #### Kelebihan, Kekurangan dan Solusi :
  - Kelebihannya adalah model dapat memprediksi gambar dengan cepat
  - Kekuranganya adalah jika gambar atau image yang kita upload tidak ada contohnya pada dataset maka kemungkinan model tidak berhasil memprediksi gambar dengan benar
  - Solusi yang ditawarkan untuk masalah diatas adalah memperbanyak dataset sesuai dengan kebutuhan model atau bisa juga menggunakan image augmentation
 
 ## Evaluation
 ### Hasil Grafik Plot Model  :
  - Plot Loss
  
  ![2 plot](https://user-images.githubusercontent.com/111211477/201364803-166bd23c-d662-4140-9117-f1bf0800ff04.png)
  
  - Plot Accuracy
  
  ![2 plott](https://user-images.githubusercontent.com/111211477/201365270-f1e48da2-a077-4cf9-a098-024c0a402a0e.png)
  
 
 ### Hasil Prediksi :
 
 ![blubbery](https://user-images.githubusercontent.com/111211477/201365850-bf0b69db-e8e6-4663-be29-7ead74d20bd2.png)
 
Metrik digunakan adalah akurasi dimana akurasi merupakan metode pengujian berdasarkan tingkat kedekatan antara nilai prediksi dengan nilai aktual. Dengan mengetahui jumlah data yang diklasifikasikan secara benar maka dapat diketahui akurasi hasil prediksi. Dapat anda lihat bahwa akurasi saya overfit namun model masih dapat memprediksi gambar dengan benar.
 
 Analisa hasil dari metrik akurasi dan grafik plot :
  - loss : adalah training loss. Yaitu nilai dari penghitungan loss function dari training dataset dan prediksi dari model. Loss yang dihasilkan dari model ini dimulai dari 4.9733 dan terus turun sampai 2.7213e-05 dimana tidak lebih dari 1%.
  - accuracy : adalah training accuracy. Yaitu nilai dari penghitungan akurasi dari training dataset dan prediksi dari model. Accuracy yang dihasilkan dari model yang dibuat adalah sebesar 0.58 sampai 1.000 atau setara dengan 58% - 100%.
  - val_loss : adalah nilai penghitungan loss function dari validation dataset dan prediksi dari model dengan input data dari validation dataset. Val_loss yang dihasilkan dari model sebesar 0.8013 atau 80% nilai ini turun sampai 20% kemudian naik lagi menjadi 44%.
  - val_accuracy : adalah nilai penghitungan akurasi dari validation dataset dan prediksi dari model dengan input data dari validation dataset. Val_accuracy yang dihasilkan dari model yang dibuat adalah sebesar 0.79 sampai 0.96 atau setara dengan 79% - 96%, dengan nilai akhir sebesar 82%.
 #### Kesimpulan :
 Epoch yang akan ditampilkan adalah 25 dan fungsi evaluate mengembalikan 2 nilai. Yang pertama jika dilihat dari Plot Loss, training terus menurun sedangkan validasi menurun ke suatu titik dan mulai meningkat lagi. Ini merupakan ciri bahwa model mengalami *overfit*. Yang kedua adalah nilai akurasi dari model yang dibuat untuk traing mencapai 100% sedangkan validasi mencapai 82%. Dari hasil grafik plot juga dapat dilihat bahwa model yang dibuat hanya mengalami sedikit loss dan akurasi dari model yang dibuat juga sangat tinggi. Ketika gambar diinput, model dapat memprediksi ketiga gambar dengan benar. Dan hasil prediksinya dapat saya simpulkan 100%.
 
## References
N. Rondán, J. Fernández-Palleiro, R. Salveraglio, M. E. Rodríguez-Rimoldi, N. Ferro and R. Sotelo, "Self-Checkout System Prototype for Point-of-Sale using Image Recognition with Deep Neural Networks," *IEEE URUCON*, pp. 217-222, 2021.
  
