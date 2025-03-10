import numpy as np
import cv2
import glob

# Membaca gambar dari direktori
paths = glob.glob('uts/*.jpg')  # ganti sesuai direktori
list_image = []

cv2.ocl.setUseOpenCL(False)

for image_path in paths:
    img = cv2.imread(image_path)
    if img is not None:
        list_image.append(img)

# Membuat objek image stitcher
imageStitcher = cv2.Stitcher_create()
status, stitched_img = imageStitcher.stitch(list_image)

# Memeriksa apakah stitching berhasil
if status == cv2.Stitcher_OK:
    # Konversi image ke grayscale
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 1)  # Masking black area

    # gabung mask dengan kernel untuk menutupi area hitam
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)

    # lakukan inpainting untuk mengisi area hitam
    inpainted_img = cv2.inpaint(stitched_img, mask_dilated, 5, cv2.INPAINT_TELEA)

    # Simpan hasil
    cv2.imwrite("uts/hasil/hasil_inpainted.jpg", inpainted_img)  # ganti sesuai direktori
    print("Hasil berhasil disimpan di uts/hasil/hasil_inpainted.jpg")
else:
    print(f"Stitching gagal dengan status kode: {status}")