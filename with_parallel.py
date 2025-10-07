import cv2
import numpy as np
import time
import math
from collections import Counter
import concurrent.futures
import sys

def preprocess_image(image_path, blur_ksize=(5, 5)):
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise ValueError(f"Image at {image_path} not found.")
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    _, binary_img = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray_blurred, original_img, gray, binary_img

def detect_circles(img, gray_blurred, questions, options, mode_radius, a_t, b_t):
    upper_a = max(a_t, mode_radius)
    lower_a = min(a_t, mode_radius)
    upper_a_threshold = math.ceil(upper_a + upper_a * 0.20)
    lower_a_threshold = math.floor(lower_a - lower_a * 0.20)

    for b in range(max(b_t, 0), max(b_t - 10, 0), -1):
        for a in range(max(upper_a_threshold, 0), max(lower_a_threshold, 0) - 1, -1):
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, a, 
                            param1=b, param2=3*b, minRadius=a//2, maxRadius=a)
            if detected_circles is not None:
                print(f"detect_circles: {detected_circles.shape[1]}, a: {a}, b: {b}")
                if detected_circles.shape[1] == questions * options:
                    detected_circles = np.uint16(np.around(detected_circles))
                    return detected_circles
    cv2.destroyAllWindows()
    return None

def hough_for_params(params):
    a, b, gray_blurred, questions, options = params
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, a,
                            param1=b, param2=3*b, minRadius=a//2, maxRadius=a)
    if detected_circles is not None and detected_circles.shape[1] == questions * options:
        detected_circles = np.uint16(np.around(detected_circles))
        radii_list = [pt[2] for pt in detected_circles[0, :]]
        most_common_radius = Counter(radii_list).most_common(1)[0][0]
        return detected_circles, most_common_radius, a, b
    return None

def detect_circles_template_parallel(img, gray_blurred, questions, options):
    param_combinations = [(a, b, gray_blurred, questions, options)
                          for b in range(15, 5, -1)
                          for a in range(5, 35)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        results = list(executor.map(hough_for_params, param_combinations))

    for result in results:
        if result is not None:
            detected_circles, most_common_radius, a, b = result
            print(f"circles:{detected_circles.shape[1]}, a:{a}, b:{b}")
            print(f"Most common radius: {most_common_radius}")
            return detected_circles, most_common_radius, a, b

    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    questions = 40
    options = 4

    # Preprocess images
    start_1 = time.time()
    gray_blurred_s, original_img_s, gray_s, binary_img_s = preprocess_image('./Test/7_S.jpg')
    gray_blurred_t, original_img_t, gray_t, binary_img_t = preprocess_image('./Test/7_T.jpg')
    gray_blurred_k, original_img_k, gray_k, binary_img_k = preprocess_image('./Test/7_K.jpg')
    end_1 = time.time()
    print("Preprocessing time:", end_1 - start_1)

    # Detect circles template (parallel)
    start_2 = time.time()
    detected_circles_t, mode_radius, a_t, b_t = detect_circles_template_parallel(
        original_img_t, gray_blurred_t, questions, options
    )
    end_2 = time.time()
    print("Detecting circles time template (parallel):", end_2 - start_2)

    # Detect circles other images using template params
    start_3 = time.time()
    detected_circles_s = detect_circles(
        original_img_s, gray_blurred_s, questions, options, mode_radius, a_t, b_t
    )
    detected_circles_k = detect_circles(
        original_img_k, gray_blurred_k, questions, options, mode_radius, a_t, b_t
    )
    end_3 = time.time()
    print("Detecting circles time:", end_3 - start_3)
