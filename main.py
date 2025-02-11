from cerionta import calculate_psnr
import json
from preprocess import preprocess_image

import cv2

if __name__ == "__main__":
    img_original = cv2.imread('../train/000000000139.jpg')
    img_compressed = cv2.imread('../train_noisy/000000000139.jpg')

    f = open('annotations_train.json')
    train = json.load(f)
    g = open('annotations_val.json')
    test = json.load(g)

    # Calculați PSNR
    psnr_value = calculate_psnr(img_original, img_compressed)
    print("PSNR:", psnr_value)

    img_compressed = preprocess_image(img_compressed)

    print("SDadsad")
