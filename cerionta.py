import numpy as np

def calculate_psnr(img1, img2):
    # Verificați dacă imaginile au aceeași dimensiune
    if img1.shape != img2.shape:
        raise ValueError("Imaginile trebuie să aibă aceeași dimensiune")

    # Calculează diferența pătratică
    diff = (img1.astype(np.float32) - img2.astype(np.float32)) ** 2

    # Calculează eroarea medie pătratică (MSE)
    mse = np.mean(diff)

    # Calculează PSNR
    max_pixel_value = 255.0  # Valoarea maximă pe care o poate avea un pixel într-o imagine de 8 biți
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr


