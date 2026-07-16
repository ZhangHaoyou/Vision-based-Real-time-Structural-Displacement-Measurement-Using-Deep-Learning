import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lines_and_points(img):
    """
    Detect and return the four points:
    - top-left intersection, top-right intersection
    - bottom-left, bottom-right
    Additionally, mark these points and draw the bounding box.
    """
    if len(img.shape) == 2:  # Image is grayscale
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert back to color image for visualization purposes (to draw red lines)
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    # Use HoughLinesP to detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is None:
        print("No lines detected.")
        return None, img_color

    # Find the longest 3 lines by length
    longest_lines = sorted(lines, key=lambda x: np.linalg.norm([x[0][0] - x[0][2], x[0][1] - x[0][3]]), reverse=True)[:3]

    # Identify vertical and horizontal lines
    vertical_lines = []
    horizontal_line = None

    for line in longest_lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < abs(y1 - y2):  # Vertical line
            vertical_lines.append(line)
        else:  # Horizontal line
            horizontal_line = line

    # Extend vertical lines upwards and downwards
    extended_vertical_lines = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line[0]
        y1, y2 = min(y1, y2), max(y1, y2)  # Ensure proper vertical direction
        extended_vertical_lines.append((x1, y1, x2, y2))

    # Ensure both vertical lines are of equal length
    left_line = extended_vertical_lines[0]
    right_line = extended_vertical_lines[1]

    # Determine the longer and shorter lines
    left_length = left_line[3] - left_line[1]
    right_length = right_line[3] - right_line[1]

    if left_length < right_length:
        # Extend the left vertical line downwards to match the right line's length
        left_line = (left_line[0], left_line[1], left_line[2], right_line[3])
    else:
        # Extend the right vertical line downwards to match the left line's length
        right_line = (right_line[0], right_line[1], right_line[2], left_line[3])

    # Extend horizontal line leftwards and rightwards
    if horizontal_line is not None:
        x1, y1, x2, y2 = horizontal_line[0]

    # Find and return the intersection points and bottom points
    left_intersection = (left_line[0], y1)
    right_intersection = (right_line[0], y1)
    bottom_left = (left_line[0], left_line[3])
    bottom_right = (right_line[0], right_line[3])

    points = {
        "top_left": left_intersection,
        "top_right": right_intersection,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right
    }

    return points, img_color

def apply_perspective_transform(img, points):
    """Apply a perspective transform using the detected rectangle points."""
    top_left = points["top_left"]
    top_right = points["top_right"]
    bottom_left = points["bottom_left"]
    bottom_right = points["bottom_right"]

    # Compute the width and height of the new perspective-corrected image
    width = int(np.linalg.norm(np.array(top_right) - np.array(top_left)))
    height = int(np.linalg.norm(np.array(bottom_left) - np.array(top_left)))

    # Define the destination points (rectangle corners) for the transformation
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Define the source points as the detected four corners
    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to the image
    transformed_img = cv2.warpPerspective(img, matrix, (width, height))

    return transformed_img, matrix

def apply_existing_transform(full_img, matrix):
    """
    Apply the perspective transformation to the full image using the same transformation matrix.
    """
    h, w = full_img.shape[:2]

    # Apply the same transformation matrix to the full image
    transformed_full_img = cv2.warpPerspective(full_img, matrix, (w, h))

    return transformed_full_img

def correct_perspective(image_path):
    """
    Correct the perspective of an image by detecting a skewed rectangle and applying a perspective transform.
    """
    # Load the full color image
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Perform opening and closing operations to clean up the image
    ret, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11, 11), np.uint8)
    img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    img_clear = img_closing

    # Manually crop a small region to detect lines (similar to target.jpg)
    margin = 400
    target_box = [995-margin, 1140+margin, 1530-margin, 1661+margin]
    target = img_clear[target_box[0]:target_box[1], target_box[2]:target_box[3]]

    # Detect the points in the cropped region
    points, img_with_points = detect_lines_and_points(target)

    # Show the image with marked points and bounding box
    if points:
        # Apply the perspective transform to the target region (grayscale)
        transformed_target, matrix = apply_perspective_transform(
            img_color[target_box[0]:target_box[1], target_box[2]:target_box[3]], points
        )

        # Show the perspective-corrected target image
        cv2.imshow('Perspective Corrected Target', transformed_target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply the same transform to the full color image
        transformed_full_img = apply_existing_transform(img_color, matrix)

        # Show the perspective-corrected full image
        cv2.imshow('Perspective Corrected Full Image', transformed_full_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return transformed_full_img

    return None

if __name__ == '__main__':
    corrected_image = correct_perspective('./experimental_data/test.png')
