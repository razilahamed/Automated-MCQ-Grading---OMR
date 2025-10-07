import streamlit as st
import numpy as np
from omr_core import *

st.title("Automated MCQ Grading (OMR)")

questions = st.number_input("Number of Questions", min_value=1, value=40)
options = st.number_input("Options per Question", min_value=2, value=4)

st.header("Upload Images")
template_file = st.file_uploader("Template Sheet", type=["jpg", "png"])
key_file = st.file_uploader("Answer Key Sheet", type=["jpg", "png"])
student_file = st.file_uploader("Student Sheet", type=["jpg", "png"])

if template_file and student_file and key_file:
    st.success("All images uploaded. Click 'Grade' to start.")
    if st.button("Grade"):
        # Preprocess images
        gray_blurred_t, original_img_t, gray_t, binary_img_t = preprocess_image(template_file.read())
        gray_blurred_s, original_img_s, gray_s, binary_img_s = preprocess_image(student_file.read())
        gray_blurred_k, original_img_k, gray_k, binary_img_k = preprocess_image(key_file.read())

        # Detect circles in template
        detected_circles_t, mode_radius, a_t, b_t = detect_circles_template(gray_blurred_t, questions, options)
        if detected_circles_t is None:
            st.error("Template circles not detected.")
            st.stop()
        detected_circles_s = detect_circles(gray_blurred_s, questions, options, mode_radius, a_t, b_t)
        detected_circles_k = detect_circles(gray_blurred_k, questions, options, mode_radius, a_t, b_t)
        if detected_circles_s is None or detected_circles_k is None:
            st.error("Student or key circles not detected.")
            st.stop()

        # Pixel counting
        circles_s = detected_circles_s[0, :]
        circles_t = detected_circles_t[0, :]
        circles_k = detected_circles_k[0, :]
        
        # Draw and display detected circles for each sheet only once
        # Template sheet
        img_template = original_img_t.copy()
        for (x, y, r) in circles_t:
            cv2.circle(img_template, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(img_template, (int(x), int(y)), 2, (0, 0, 255), 3)
        img_template_rgb = cv2.cvtColor(img_template, cv2.COLOR_BGR2RGB)
        st.image(img_template_rgb, caption="Template Sheet (Detected Circles)", channels="RGB")

        # Key sheet
        img_key = original_img_k.copy()
        for (x, y, r) in circles_k:
            cv2.circle(img_key, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(img_key, (int(x), int(y)), 2, (0, 0, 255), 3)
        img_key_rgb = cv2.cvtColor(img_key, cv2.COLOR_BGR2RGB)
        st.image(img_key_rgb, caption="Key Sheet (Detected Circles)", channels="RGB")

        # Student sheet
        img_student = original_img_s.copy()
        for (x, y, r) in circles_s:
            cv2.circle(img_student, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(img_student, (int(x), int(y)), 2, (0, 0, 255), 3)
        img_student_rgb = cv2.cvtColor(img_student, cv2.COLOR_BGR2RGB)
        st.image(img_student_rgb, caption="Student Sheet (Detected Circles)", channels="RGB")
        
        pixel_counts_s = count_black_and_white_pixels(binary_img_s, circles_s)
        pixel_counts_t = count_black_and_white_pixels(binary_img_t, circles_t)
        pixel_counts_k = count_black_and_white_pixels(binary_img_k, circles_k)

        # Group by row
        grouped_rows_s = group_by_row_dbscan(pixel_counts_s, eps=10)
        grouped_rows_t = group_by_row_dbscan(pixel_counts_t, eps=10)
        grouped_rows_k = group_by_row_dbscan(pixel_counts_k, eps=10)

        # Calculate max SD from template
        stat_template = mean_sd(grouped_rows_t, options)
        max_tuple = max(stat_template, key=lambda x: x[1])
        max_sd = max_tuple[1]

        # Detect marked bubbles
        answer_student = detect_marked_and_unmarked_bubbles(grouped_rows_s, options, deviation_threshold=max_sd)
        answer_key = detect_marked_and_unmarked_bubbles(grouped_rows_k, options, deviation_threshold=max_sd)

        # Score
        total_score, result_per_question = calculate_score(answer_key, answer_student)
        st.header(f"Total Score: {total_score} / {questions}")
        # st.write("Per-question results:")
        # for idx, result in enumerate(result_per_question, start=1):
        #     status = "✅ Correct" if result == 1 else "❌ Incorrect"
        #     st.write(f"Q{idx}: {status}")