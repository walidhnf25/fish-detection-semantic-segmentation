# Segmentasi Semantik Ikan dengan DeepLabv3 dan MobileNetV3

Proyek ini memisahkan piksel ikan (foreground) dari latar belakang pada gambar akuarium (segmentasi biner). Model yang dipakai adalah DeepLabv3 dengan backbone MobileNetV3-Large (PyTorch/torchvision) yang dilengkapi decoder dengan skip connection. Backbone ringan dipilih agar model cocok untuk perangkat dengan sumber daya terbatas.

## Hasil

Evaluasi pada data validasi (20% dari 390 gambar), dengan model terbaik setelah 30 epoch:

| Metrik | Nilai |
|---|---|
| Dice Score | 0.6853 |
| IoU | 0.5605 |
| Pixel Accuracy | 0.9550 |
| Precision | 0.7286 |
| Recall | 0.6879 |
| F1 Score | 0.6853 |
| Waktu inferensi rata-rata | 379.52 ms/gambar |

- Confusion matrix piksel (persen dari total piksel): TN 89.912%, TP 5.588%, FN 2.910%, FP 1.590%.
- Precision lebih tinggi dari recall dan FN hampir dua kali FP, jadi model cenderung konservatif: lebih sering melewatkan piksel ikan daripada salah menandai latar.
- Loss training turun dari sekitar 0.62 ke 0.08, sedangkan loss validasi turun dari 0.47 ke sekitar 0.28 lalu stabil. Selisih keduanya menandakan overfitting ringan.
- Sekitar 65% gambar validasi punya Dice di atas 0.7, dengan konsentrasi terbanyak di 0.8 sampai 0.9. Sebagian kecil gambar berada di 0.2 sampai 0.4.

Angka di atas adalah hasil eksperimen penulis. Menjalankan ulang notebook dapat memberi hasil yang sedikit berbeda karena split data, seed, versi library, dan hardware (terutama waktu inferensi).

## Alur Kerja

1. **Hubungkan dataset** dari Google Drive dan pasangkan gambar dengan mask berdasarkan nama file.
2. **Preprocessing**:
   - Resize ke 512x512 (bilinear untuk gambar, nearest untuk mask)
   - Normalisasi ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
   - Mask ID kelas menjadi tensor biner `[N, 1, H, W]` (semua ID di atas 0 menjadi ikan)
   - Tanpa augmentasi
   - Split 80% training dan 20% validasi, terstratifikasi berdasarkan proporsi piksel ikan pada mask
3. **Model**:
   - Backbone MobileNetV3-Large, pretrained ImageNet
   - ASPP dengan laju dilasi `[6, 12, 18]`, global average pooling, dan konvolusi 1x1
   - Decoder: upsampling bilinear, skip connection dari fitur level rendah backbone, konvolusi refinement
   - Output: konvolusi 1x1 dengan 1 kanal, sigmoid untuk probabilitas piksel
4. **Training**:
   - Adam (lr 1e-4, betas 0.9/0.999, weight decay 1e-6), batch 4, 30 epoch
   - Loss `alpha * BCE + (1 - alpha) * Dice loss` dengan `alpha = 0.5`
   - Checkpoint dengan Dice validasi tertinggi disimpan ke Google Drive
5. **Evaluasi**: Dice, IoU, Pixel Accuracy, Precision, Recall, F1 (per gambar lalu dirata-ratakan), histogram metrik, confusion matrix, dan waktu inferensi.
6. **Pengujian visual**: kurva training, kasus terbaik dan terburuk (gambar, mask asli, prediksi, overlay), heatmap aktivasi layer, serta fungsi `predict_image` untuk gambar baru.

## Dataset

Dataset **tidak disertakan** di repository ini. Proyek memakai [Aquarium Fish Semantic Segmentation Dataset](https://www.kaggle.com/datasets/bemorekgg/aquarium-fish-semantic-segmentation-dataset) oleh kildongGo di Kaggle (lisensi Apache 2.0). Folder `segment_semantic` berisi 391 pasangan gambar dan mask dengan resolusi bervariasi. Mask dibuat dengan Segment Anything Model (SAM) memakai bounding box sebagai prompt, bukan digambar manual.

Struktur asli dataset:

```
aquarium_pretrain/
├── segment_semantic/
│   ├── images/        # gambar asli          <- dipakai
│   ├── masks/         # mask ID kelas        <- dipakai
│   ├── color_masks/   # mask berwarna untuk visualisasi (tidak dipakai)
│   └── overlay/       # gambar + mask transparan (tidak dipakai)
├── train/             # format YOLO (tidak dipakai)
└── test/              # format YOLO (tidak dipakai)
```

Hanya `images/` dan `masks/` dari `segment_semantic` yang dibutuhkan. Unduh dataset dari Kaggle, lalu unggah kedua folder itu ke Google Drive dengan struktur berikut (`/content/drive/MyDrive/dataset_ikan`):

```
dataset_ikan/
├── images/    # isi segment_semantic/images
└── masks/     # isi segment_semantic/masks
```

Mask berisi ID kelas (0 = latar belakang, 1 dan seterusnya = kelas objek). Notebook menggabungkan semua ID di atas 0 menjadi satu kelas "ikan" sehingga tugasnya segmentasi biner. Gambar dan mask dipasangkan berdasarkan nama file tanpa ekstensi. Jika nama mask punya akhiran (mis. `ikan01_mask.png`), isi `MASK_SUFFIX = '_mask'` di Cell 2.

## Cara Menjalankan

Notebook ditulis untuk **Google Colab** karena memakai `google.colab.drive`. Colab sudah menyediakan PyTorch, torchvision, dan Matplotlib, jadi tidak perlu instalasi tambahan.

1. Unggah dataset ke Google Drive sesuai struktur di atas.
2. Buka [fish-segmentation.ipynb](fish-segmentation.ipynb) di Google Colab.
3. Aktifkan GPU: `Runtime > Change runtime type > T4 GPU`.
4. Jalankan semua cell berurutan. Izinkan akses Google Drive saat diminta. Bobot ImageNet untuk backbone diunduh otomatis saat Cell 4.

Untuk menjalankan secara lokal, pasang dependensi (untuk GPU NVIDIA, pasang PyTorch sesuai [pytorch.org](https://pytorch.org/get-started/locally/) lebih dulu), letakkan folder `dataset_ikan` di samping notebook, lalu buka notebook:

```bash
pip install -r requirements.txt
jupyter notebook
```

Di luar Colab, Cell 2 otomatis membaca dataset dan menyimpan keluaran di folder kerja saat ini.

Keluaran training disimpan ke Google Drive di `fish_segmentation_output/`: `best_model.pt` (bobot terbaik + konfigurasi) dan `history.json` (loss dan metrik per epoch). Keduanya tidak termasuk dalam repository.

## Teknologi

- Python, NumPy, Pillow
- PyTorch dan torchvision (MobileNetV3-Large, ASPP)
- Matplotlib

## Keterbatasan

- **Data validasi juga dipakai untuk memilih checkpoint terbaik** dan sebagai laporan akhir, sehingga hasilnya cenderung sedikit optimistis. Untuk laporan yang lebih ketat, sisihkan test set terpisah.
- **Waktu inferensi** diukur dengan batch 1 dan hanya forward pass model (tanpa praproses), sehingga bergantung pada hardware.
- **Dataset kecil** (sekitar 390 gambar) dan hanya segmentasi biner (ikan vs latar), tidak membedakan spesies.
- **Mask dibuat otomatis oleh SAM**, bukan anotasi manual, sehingga batas objek pada mask bisa mengandung kesalahan yang ikut memengaruhi training dan metrik.
- **Tanpa augmentasi.** Performa menurun pada gambar dengan pencahayaan ekstrem (terlalu gelap atau terang), ikan tertutup dekorasi akuarium, sirip atau tubuh yang transparan, pose atau sudut pandang yang jarang, dan gambar buram karena gerakan.

## Struktur Repository

```
.
├── fish-segmentation.ipynb   # notebook utama (Google Colab)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Ucapan Terima Kasih

Dataset: [Aquarium Fish Semantic Segmentation Dataset](https://www.kaggle.com/datasets/bemorekgg/aquarium-fish-semantic-segmentation-dataset) oleh kildongGo (Kaggle, Apache 2.0).

## Lisensi

Kode dalam repository ini dirilis di bawah [Lisensi MIT](LICENSE).
