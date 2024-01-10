# -*- coding: utf-8 -*-
"""Revisi 7 Submission ML Terapan 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q_JBxIOGVSdxY3Xf4y7QmLDRZuTlYol_

Mengecek Versi dari TF yang digunakan
"""

import tensorflow as tf
print(tf.__version__)

"""Mengimpor library-library yang akan digunakan dalam pembuatan model ini"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.optimizers import RMSprop

"""Import zipfile digunakan untuk mengambil file bertipe zip"""

import zipfile, os

"""Pada kode dibawah ini adalah untuk melakukan ekstraksi terhadap dataset yang diunduh yaitu  dataset dalam bentuk file zip"""

local_zip = 'Revisi Dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close() 
base_dir = 'Revisi Dataset'

"""Fungsi listdir dari kelas os untuk melihat direktori yang terdapat pada dataset."""

os.listdir(base_dir)

"""Membagi data menjadi training dan validation sesuai dengan folder yang telah dibuat, juga menampilkan jumlah gambar serta class yang telah dibagi didalam folder tersebut. Traing berfungsi untuk melatih model sedangkan validasi digunakan untuk proses validasi model dan mencegah overfitting

pada program dibawah ini berfungsi  mendefinisikan terlebih dahulu bagaimana transformasi data yang akan digunakan.  Pada perintah di bawah ini program akan melakukan Rescaling data menjadi 1/255, Shearing image skala 0.2, Zooming image dengan range 0.2, dan melakukan Vertical flip. Setelah itu mendefinisikan generatornya darimana sumber datanya berasal dengan  menggunakan data yang berasal dari folder (dictionary).




> pada code dibawah,  menggunakan flow from dir dimana fungsi tersebut mengarahkan kepada folder set yang telah dibuat sebelumnya. di mana :


1.   target size = dimensi dari citra yang akan digunakan dalam proses training
2.  class mode = metode pemilihan klasifikasi
"""

TRAINING_DIR = 'Revisi Dataset/train'
VALIDATION_DIR = 'Revisi Dataset/test'
 
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    fill_mode='nearest',
)
 
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
 
validation_datagen = ImageDataGenerator(
    rescale=1.0/255
)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                        class_mode='categorical',
                                                        target_size=(150, 150))

os.listdir(TRAINING_DIR)

os.listdir(VALIDATION_DIR)

"""Pada code dibawah ini membuat  arsitektur model  menggunakan 3 lapis convolution yang berfungsi untuk mengekstraksi atribut pada gambar, dan 2 hidden layer dengan 512 buah unit perseptron. Sedangkan layer max pooling berguna untuk mereduksi resolusi gambar

Digunakan juga 2 dropout layer untuk mencegah overfiting dari model yang dibuat dengan besar 50% dan 20% yang di dropout
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.Dropout(0.8),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

"""code dibawah ini adalah menyusun model menjadi siap dilakukan proses training. Dimana variabel yang digunakan sebagai berikut

1. Optimizer = merupakan metode optimasi yang digunakan.
2. loss = adalah metode pengukuran nilai loss berdasarkan pada nilai apa. karena Membuat flow datanya maka menggunakan categorical sehingga pada nilai loss ini juga menggunakan categorical loss.
3. Metrics = Nilai matriks yang digunakan adalah nilai akurasi sebagai nilai pengukurannya
"""

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])

"""Mengevaluasi model dan melihat apakah model underfit atau overfit. Untuk melihat loss dan akurasi model pada data test, gunakan fungsi evaluate pada model. Fungsi Evaluate mengembalikan 2 nilai. Yang pertama adalah nilai loss, dan yang kedua adalah nilai akurasinya."""

num_epochs=25
history = model.fit(
      train_generator,
      steps_per_epoch=25,# berapa batch yang akan dieksekusi pada setiap epoch
      epochs=num_epochs,#epochs dapat ditambahkan jika akurasi model belum optimal
      validation_data=validation_generator,# menampilkan akurasi pengujian data validasi
      validation_steps=5,# berapa batch yang akan dieksekusi pada setiap epoch
      verbose=2)

"""Pada code dibawah ini digunakan untuk melihat keseluruhan network dengan menggunakan model.summary()"""

model.summary()

"""Proses yang terakhir adalah melihat hasil Model dengan prediksi klasifikasi yaitu ketika program  dijalankan kemudian dimasukkan gambar baru, maka akan menunjukkan hasil pengklasifikasian berdasarkan ciri-ciri atau kriteria tertentu yang sudah dilatih sebelumnya menggunakan data train. Berikut adalah hasilnya"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import keras
from keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

uploaded = files.upload()
 
for fn in uploaded.keys():
 
  # predicting images
  path = fn
  img = keras.utils.load_img(path, target_size=(150,150))
  imgplot = plt.imshow(img)
  x = keras.utils.img_to_array(img)
  x = np.expand_dims(x, axis=0)
 
  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  
  print(fn)
  if classes[0][0]==1:
    print('Banana')
  elif classes[0][1]==1:
    print('Avocado')
  elif classes[0][2]==1:
    print('Blubbery')
  else:
    print('Tidak diketahui')

plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, num_epochs), history.history["loss"], label="training")
plt.plot(np.arange(0, num_epochs), history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(np.arange(0, num_epochs), history.history["accuracy"], label="training")
plt.plot(np.arange(0, num_epochs), history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()