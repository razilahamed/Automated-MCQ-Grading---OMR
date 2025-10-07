import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter, defaultdict

def preprocess_image(image_bytes, blur_ksize=(5, 5)):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    _, binary_img = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray_blurred, original_img, gray, binary_img

def detect_circles_template(gray_blurred, questions, options):
    for b in range(15, 5, -1):
        for a in range(5, 35):
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, a,
                                                param1=b, param2=3*b, minRadius=a//2, maxRadius=a)
            if detected_circles is not None and detected_circles.shape[1] == questions * options:
                detected_circles = np.uint16(np.around(detected_circles))
                radii_list = [pt[2] for pt in detected_circles[0, :]]
                most_common_radius = Counter(radii_list).most_common(1)[0][0]
                return detected_circles, most_common_radius, a, b
    return None, None, None, None

def detect_circles(gray_blurred, questions, options, mode_radius, a_t, b_t):
    import math
    upper_a = max(a_t, mode_radius)
    lower_a = min(a_t, mode_radius)
    upper_a_threshold = math.ceil(upper_a + upper_a * 0.20)
    lower_a_threshold = math.floor(lower_a - lower_a * 0.20)
    for b in range(max(b_t, 0), max(b_t - 10, 0), -1):
        for a in range(max(upper_a_threshold, 0), max(lower_a_threshold, 0) - 1, -1):
            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, a,
                                                param1=b, param2=3*b, minRadius=a//2, maxRadius=a)
            if detected_circles is not None and detected_circles.shape[1] == questions * options:
                detected_circles = np.uint16(np.around(detected_circles))
                return detected_circles
    return None

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

def group_by_row_dbscan(pixel_counts, eps=20, fallback_eps=10):
    y_coords = np.array([[y] for x, y, r, bc, wc in pixel_counts])
    def _cluster(eps_val):
        labels = DBSCAN(eps=eps_val, min_samples=1).fit(y_coords).labels_
        groups = defaultdict(list)
        for lbl, data in zip(labels, pixel_counts):
            groups[lbl].append(data)
        sorted_rows = sorted(groups.values(), key=lambda row: np.mean([c[1] for c in row]))
        return [sorted(row, key=lambda c: c[0]) for row in sorted_rows]
    rows = _cluster(eps)
    if len({len(r) for r in rows}) != 1:
        rows = _cluster(fallback_eps)
    return rows

def mean_sd(grouped_rows, num_options):
    temp = []
    for row in grouped_rows:
        for question_start in range(0, len(row), num_options):
            question_group = row[question_start:question_start + num_options]
            white_pixel_values = [white for (_, _, _, _, white) in question_group]
            mean_white = np.mean(white_pixel_values)
            std_dev_white = np.std(white_pixel_values)
            temp.append((mean_white, std_dev_white))
    return temp

def detect_marked_and_unmarked_bubbles(grouped_rows, num_options, deviation_threshold):
    import numpy as np
    marked_bubbles = []
    for row in grouped_rows:
        for question_start in range(0, len(row), num_options):
            question_group = row[question_start:question_start + num_options]
            white_pixel_values = [white for (_, _, _, _, white) in question_group]
            std_dev_white = np.std(white_pixel_values)
            if std_dev_white < deviation_threshold:
                marked_bubbles.append([0] * num_options)
            else:
                marked_question = np.zeros(num_options, dtype=int)
                min_white_index = np.argmin(white_pixel_values)
                marked_question[min_white_index] = 1
                marked_bubbles.append(marked_question)
    return np.array(marked_bubbles)

def calculate_score(answer_key, answer_student):
    import numpy as np
    total_score = 0
    result_per_question = []
    for key, student in zip(answer_key, answer_student):
        if np.array_equal(key, student):
            total_score += 1
            result_per_question.append(1)
        else:
            result_per_question.append(0)
    return total_score, result_per_question