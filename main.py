from cerionta import calculate_psnr;

import cv2

if __name__ == "__main__":
    img_original = cv2.imread('../train/000000000139.jpg')
    img_compressed = cv2.imread('../train_noisy/000000000139.jpg')

    # Calculați PSNR
    psnr_value = calculate_psnr(img_original, img_compressed)
    print("PSNR:", psnr_value)