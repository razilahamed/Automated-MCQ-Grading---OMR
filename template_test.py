import os
import cv2
import json
import numpy as np
from collections import Counter

# Image preprocessing step
def preprocess_image(image_path, blur_ksize=(5, 5)):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Error: File does not exist.")
    elif os.path.getsize(image_path) == 0:
        raise ValueError("Error: File is empty.")
    
    # Read the image.
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise ValueError(f"Image at {image_path} not found.")
    
    # Convert to grayscale.
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur.
    gray_blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    
    # Convert the image to a binary image.
    _, binary_img = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "original_img": original_img,
        "gray": gray,
        "gray_blurred": gray_blurred,
        "binary_img": binary_img
    }

# Circle detection step
def detect_circles_template(gray_blurred, questions, options):
    for b in range(15, 5, -1):
        for a in range(5, 35):
            detected_circles = cv2.HoughCircles(
                gray_blurred, cv2.HOUGH_GRADIENT, 1, a,
                param1=b, param2=3 * b,
                minRadius=a // 2, maxRadius=a
            )
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))
                if detected_circles.shape[1] == questions * options:
                    radii_list = [pt[2] for pt in detected_circles[0, :]]
                    most_common_radius = Counter(radii_list).most_common(1)[0][0]
                    return detected_circles, most_common_radius, a, b
    return None, None, None, None

# Count black and white pixels in circles
def count_black_and_white_pixels(img, circles):
    counts = []
    for (x, y, r) in circles:
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, (255), thickness=-1)
        masked_area = cv2.bitwise_and(img, mask)
        black_pixels = np.sum(masked_area == 0)
        white_pixels = np.sum(masked_area == 255)
        counts.append((x, y, r, black_pixels, white_pixels))
    return counts

# Save pixel count data as JSON
def save_pixel_counts_to_json(pixel_counts, output_file):
    pixel_counts_as_int = [tuple(map(int, tpl)) for tpl in pixel_counts]
    with open(output_file, 'w') as f:
        json.dump(pixel_counts_as_int, f, indent=4)

# Main script
def main():
    image_path = "./Test/f7.jpg"  
    questions = 40  
    options = 4  
    output_file = "pixel_counts_f7.json"

    try:
        # Preprocess the image
        processed_images = preprocess_image(image_path)
        binary_img = processed_images["binary_img"]
        gray_blurred = processed_images["gray_blurred"]

        # Detect circles
        detected_circles, most_common_radius, a, b = detect_circles_template(
            gray_blurred, questions, options
        )
        if detected_circles is None:
            raise ValueError("Failed to detect the required number of circles.")
        
        # Count black and white pixels in detected circles
        circles = detected_circles[0, :]
        pixel_counts = count_black_and_white_pixels(binary_img, circles)
        
        # Save results to JSON
        save_pixel_counts_to_json(pixel_counts, output_file)
        print(f"Pixel counts saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
