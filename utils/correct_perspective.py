import cv2
import numpy as np


def is_black_pixel(img, x, y, count=-1):
    """Check if a pixel at (x, y) is black."""
    if y >= 0.9*img.shape[0] and count in [427, 2761, 2817]:
        return True
    return img[y, x] == 0

def extend_line_vertically(img, x, y1, y2, horizontal_line=None, extend_down=True, count=-1):
    """Extend a vertical line up (y1) until it intersects with a horizontal or slanted line,
    and down (y2) until at least 0.85 of the image height is reached."""
    img_height = img.shape[0]
    target_down_limit = int(0.9 * img_height)  # 85% of the image height
    if horizontal_line is not None:
        # Unpack the horizontal line coordinates
        x1_h, y1_h, x2_h, y2_h = horizontal_line

        # Extend upwards until the vertical line intersects with the horizontal line
        while y1 > 0:
            # Check for intersection between vertical line (x, y1) and horizontal line
            intersection = find_intersection(x, y1, x, y1 - 1, x1_h, y1_h, x2_h, y2_h)
            if intersection is not None:
                break  # Stop if an intersection is found
            y1 -= 1
    else:
        # Extend upwards
        while y1 > 0 and not is_black_pixel(img, x, y1, count):
            y1 -= 1

    # Extend downwards
    if extend_down and (count in [427, 2427, 2761, 2817]):
        while (y2 < img_height - 1 and not is_black_pixel(img, x, y2, count)) or y2 < target_down_limit:
            y2 += 1
    elif extend_down:
        while (y2 < img_height - 1 and not is_black_pixel(img, x, y2, count)):
            y2 += 1

    return y1, y2

def extend_line_horizontally(img, y, x1, x2):
    """Extend a horizontal line left (x1) and right (x2) until encountering the extended vertical lines."""
    # Extend leftwards
    while x1 > 0 and not is_black_pixel(img, x1, y):
        x1 -= 1

    # Extend rightwards
    while x2 < img.shape[1] - 1 and not is_black_pixel(img, x2, y):
        x2 += 1

    return x1, x2

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Find the intersection point of two lines defined by (x1, y1)-(x2, y2) and (x3, y3)-(x4, y4)."""
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        print("Lines are parallel, no intersection", x1, y1, x2, y2, x3, y3, x4, y4)
        return None  # Lines are parallel, no intersection

    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return int(intersect_x), int(intersect_y)

def merge_lines(lines):
    """Merge multiple lines by connecting the top-most point to the bottom-most point."""
    if len(lines) == 1:
        return lines[0]

    # Initialize the top-most and bottom-most points
    top_y = float('inf')   # Smallest y value for the top point
    bottom_y = -float('inf')  # Largest y value for the bottom point

    # Initialize top and bottom x-coordinates
    top_x = 0
    bottom_x = 0

    # Iterate over all lines to find the top-most and bottom-most points
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # If the first point is the top-most point
        if y1 < top_y:
            top_y = y1
            top_x = x1

        # If the second point is the top-most point
        if y2 < top_y:
            top_y = y2
            top_x = x2

        # If the first point is the bottom-most point
        if y1 > bottom_y:
            bottom_y = y1
            bottom_x = x1

        # If the second point is the bottom-most point
        if y2 > bottom_y:
            bottom_y = y2
            bottom_x = x2

    # Return a new merged line from the top-most point to the bottom-most point
    return np.array([[top_x, int(top_y), bottom_x, int(bottom_y)]])



def detect_lines_and_points(img, count, debug=False):
    """
    Detect and return the four points:
    - top-left intersection, top-right intersection
    - bottom-left, bottom-right
    """
    # Check if the image is already grayscale
    if len(img.shape) == 2:  # Image has only one channel, already grayscale
        img_gray = img
    else:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 50, 200)

    # Use HoughLinesP to detect straight lines
    if count < 400:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=20, minLineLength=110, maxLineGap=50)
    elif count == 427:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=130, maxLineGap=50)
    elif count < 1100:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 180, threshold=20, minLineLength=110, maxLineGap=50)
    elif count < 1500:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=100, maxLineGap=50)
    elif count < 1946:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=30, minLineLength=115, maxLineGap=50)
    elif count < 2423:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 180, threshold=20, minLineLength=110, maxLineGap=50)
    elif count < 2817:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=20, minLineLength=110, maxLineGap=50)
    elif count < 3000:
        lines = cv2.HoughLinesP(edges, 0.5, np.pi / 360, threshold=40, minLineLength=130, maxLineGap=20)
    else:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=115, maxLineGap=50)

    if lines is None:
        print("No lines detected.")
        return None

    # Convert the image to RGB to plot colored lines
    line_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Identify vertical and horizontal lines
    vertical_lines = []
    horizontal_line = None

    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) < abs(y1 - y2):  # Vertical line
            vertical_lines.append(line)
        else:  # Horizontal line
            horizontal_line = line
    

    # Draw the lines in yellow for debugging purposes
    if debug:
        print('Number of detected lines is', len(lines))
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            if count in [2761]:
                print('2761:', x1, y1, x2, y2)
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for the longest lines
            # Add a label for the line (Line 1, Line 2, etc.)
            label = f"Line {idx + 1}"
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(line_img, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Detected lines', line_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Check if we have at least two vertical lines
    if len(vertical_lines) < 2:
        print(f"Error: Detected only {len(vertical_lines)} vertical line(s). Cannot proceed.")
        return None
    
    # Initialize lists for the vertical and horizontal lines
    left_vertical_lines = []
    right_vertical_lines = []
    horizontal_line = None

    # Separate lines into left, right, and horizontal categories
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_center_x = (x1 + x2) // 2
        img_center_x = img_gray.shape[1] // 2

        if abs(x1 - x2) < abs(y1 - y2):  # Vertical line
            if line_center_x < img_center_x:  # Left side
                left_vertical_lines.append(line)
            else:  # Right side
                right_vertical_lines.append(line)
        else:  # Horizontal or slanted line
            horizontal_line = line
            
    # Merge multiple vertical lines on the left side
    if len(left_vertical_lines) > 1:
        left_vertical_line = merge_lines(left_vertical_lines)
    else:
        left_vertical_line = left_vertical_lines[0]

    # Merge multiple vertical lines on the right side
    if len(right_vertical_lines) > 1:
        right_vertical_line = merge_lines(right_vertical_lines)
    else:
        right_vertical_line = right_vertical_lines[0]

    # Extend vertical lines upwards and downwards
    x1, y1, x2, y2 = left_vertical_line[0]
    if count == 1129:
        up_ = horizontal_line[0]
    else:
        up_ = None
    if count in [2761]:
        pass
    else:
        y1, y2 = extend_line_vertically(img_gray, x1, y1, y2, up_, extend_down=True, count=count,)
    extended_left_line = (x1, y1, x2, y2)

    x1, y1, x2, y2 = right_vertical_line[0]
    if count in [2423, 2761, 2817]:
        pass
    else:
        y1, y2 = extend_line_vertically(img_gray, x1, y1, y2, up_, extend_down=True, count=count)
    extended_right_line = (x1, y1, x2, y2)

    # Ensure both vertical lines are of equal length
    left_length = extended_left_line[3] - extended_left_line[1]
    right_length = extended_right_line[3] - extended_right_line[1]

    print(extended_left_line, extended_right_line)
    if left_length < right_length: #################################################
        print('i am here...')
        if count in [2817]:
            pass
        else:
            extended_left_line = (extended_left_line[0], extended_left_line[1], extended_left_line[2], extended_right_line[3])
    else:
        if count in [2817]:
            pass
        else:
            extended_right_line = (extended_right_line[0], extended_right_line[1], extended_right_line[2], extended_left_line[3])

    # Extend the horizontal line (even if it's slanted) leftwards and rightwards
    if horizontal_line is not None:
        x1, y1, x2, y2 = horizontal_line[0]

    # Find and return the intersection points and bottom points
    left_intersection = find_intersection(extended_left_line[0], extended_left_line[1], extended_left_line[2], extended_left_line[3],
                                        x1, y1, x2, y2)
    right_intersection = find_intersection(extended_right_line[0], extended_right_line[1], extended_right_line[2], extended_right_line[3],
                                        x1, y1, x2, y2)
    if count == 2423:
        print(left_intersection, right_intersection)

    if left_intersection and right_intersection:
        if count in [1129, 1132, 1134, 1136, 1141, 1144, 1149, 1152, 1154, 1155, 1157, 1160, 1166, 1172, 1175, 1178, 1181, 1184, 1185, 1188, 1189, 1191, 1192, 1196, 1199,
                1203, 1208, 1212, 1213, 1218, 1223, 1224, 1225, 1230, 1231, 1238, 1247, 1248, 1249, 1290, 1291, 1298,
                1300, 1306, 1307, 1308, 1309, 1313, 1314, 1315, 1320, 1321, 1333, 1334, 1338, 1339, 1342, 1343, 1346, 1349, 1350,
                1351, 1353, 1364, 1366, 1369, 1370, 1375, 1379, 1381, 1384, 1389, 1390, 1391, 1392, 1395, 1397,
                1400, 1402, 1405, 1407, 1410, 1414, 1417, 1419, 1422, 1424, 1426, 1427, 1428, 1434, 1435, 1436, 1438, 1440, 1441, 1442, 1443,
                1451, 1454, 1459, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1540, 1579, 1580, 1599,
                1610, 1640, 1641, 1643, 1644, 1646, 1647, 1649, 1650, 1655, 1657, 1658, 1660, 1670, 1675, 1682, 1685, 1688, 1690, 1692, 1693, 1695, 1699,
                1702, 1704, 1707, 1709, 1712, 1716, 1721, 1724, 1726, 1728, 1733, 1736, 1741, 1746, 1748, 1749,
                1751, 1754, 1756, 1757, 1759, 1762, 1765, 1768, 1771, 1774, 1777, 1778, 1781, 1784, 1787, 1788, 1791, 1795, 1796, 1799,
                1803, 1804, 1808, 1809, 1812, 1813, 1818, 1819, 1823, 1824, 1825, 1829, 1830, 1831, 1833, 1837, 1838, 1839, 1840, 1846, 1847, 1848,
                1850, 1851, 1853, 1865, 1867, 1868, 1869, 1870, 1871, 1872, 1874, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1898, 1899, 
                1900, 1901, 1902, 1907, 1908, 1909, 1910, 1914, 1915, 1916, 1920, 1921, 1925, 1926, 1927, 1930, 1931, 1934, 1935, 1939, 1943, 1944, 1947, 
                1951, 1954, 1955, 1957, 1958, 1960, 1962, 1964, 1967, 1970, 1971, 1973, 1976, 1980, 1982, 1984, 1985, 1987, 1990, 1992, 1996, 1998, 
                2003, 2006, 2007, 2012, 2015, 2017, 2018, 2020, 2022, 2025, 2027, 2029, 2034, 2039, 2045, 2050, 2053, 2066, 2081, 2083, 2087, 2089, 2091, 2096, 
                2112, 2137, 2233, 2243, 2249, 2254, 2272, 2285, 2290, 2292, 2297, 2299, 
                2302, 2307, 2309, 2311, 2313, 2317, 2320, 2327, 2332, 2335, 2340, 2343, 2345, 
                2356, 2358, 2359, 2362, 2364, 2367, 2373, 2377, 2380, 2383, 2387, 2393, 2394, 2397, 2398,
                2401, 2402, 2406, 2410, 2411, 2416, 2421, 2422, 2423, 427, 2434, 2436, 2442, 2443, 2444, 
                2456, 2457, 2458, 2460, 2465, 2466, 2473, 2476, 2477, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2495, 
                2504, 2505, 2511, 2512, 2517, 2518, 2523, 2528, 2532, 2537, 2545, 2548, 2549, 
                2552, 2555, 2556, 2558, 2559, 2561, 2565, 2572, 2577, 2580, 2583, 2586, 2588, 2591, 2593, 2596, 2817,
                2601, 2606, 2613, 2614, 2616, 2621, 2625, 2626, 2630, 2632, 2633, 2647, 2652, 2682, 2684, 2706, 2761,
                # 2817, 2834, 2838, 2842, 2848, 2850, 2854, 2856, 2858, 2859, 2862, 2883, 2889, 2891, 2894, 2898, 
                # 2903, 2905, 2908, 2912, 2915, 2917, 2920, 2922, 2932, 2947, 
                # 2952, 2960, 2967, 2969, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2979, 2980, 2981, 2982, 2984, 2985, 2986, 2987, 2988, 2989, 
                # 2991, 2992, 2993, 2994, 2995, 2996, 2998, 2999,
                ]:
            # bottom_left = (lines[1][0][1], lines[1][0][2])
            # bottom_right = (lines[0][0][1], lines[0][0][2])
            # if count == 2423:
            #     print(extended_left_line)
            #     print(extended_right_line)
            bottom_left = (extended_left_line[0], extended_left_line[1])
            bottom_right = (extended_right_line[0], extended_right_line[1])
            if count in [2761]:
                left_vertical_line = lines[1][0]
                x1, y1, x2, y2 = left_vertical_line
                y1, y2 = extend_line_vertically(img_gray, x1, y1, y2, up_, extend_down=True, count=count)
                extended_left_line = (x1, y1, x2, y2)
                right_vertical_line = lines[0][0]
                x1, y1, x2, y2 = right_vertical_line
                y1, y2 = extend_line_vertically(img_gray, x1, y1, y2, up_, extend_down=True, count=count)
                extended_right_line = (x1, y1, x2, y2)
                print('left', extended_left_line)
                print('right', extended_right_line)
                bottom_left = (extended_left_line[2], extended_left_line[3])
                bottom_right = (extended_right_line[2], extended_right_line[3])
                print('lower ends', bottom_left, bottom_right)
        else:
            bottom_left = (extended_left_line[0], extended_left_line[3])
            bottom_right = (extended_right_line[0], extended_right_line[3])
        
        points = {
            "top_left": left_intersection,
            "top_right": right_intersection,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right
        }

        # Visualize the extended lines and intersections
        cv2.line(line_img, (extended_left_line[0], extended_left_line[1]), (extended_left_line[2], extended_left_line[3]), (255, 0, 0), 2)  # Blue for left 255, 0, 0
        cv2.line(line_img, (extended_right_line[0], extended_right_line[1]), (extended_right_line[2], extended_right_line[3]), (255, 0, 0), 2)  # Blue for right 0, 255, 0
        
        # Marking the intersections
        cv2.circle(line_img, left_intersection, 5, (0, 0, 255), -1)  # Red for top-left intersection
        cv2.circle(line_img, right_intersection, 5, (0, 0, 255), -1)  # Red for top-right intersection
        
        # Marking the bottom points
        cv2.circle(line_img, bottom_left, 5, (0, 0, 255), -1)  # Red for bottom-left
        cv2.circle(line_img, bottom_right, 5, (0, 0, 255), -1)  # Red for bottom-right
        
        # Show the final image with lines and points
        if debug:
            cv2.imshow('Extended Lines with Intersections', line_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return points
    else:
        return None


def apply_perspective_transform(img, points):
    """Apply a perspective transform using the given four points and return both the matrix and transformed image."""
    if points is None:
        print("No valid points to apply perspective transform.")
        return img

    top_left = points["top_left"]
    top_right = points["top_right"]
    bottom_left = points["bottom_left"]
    bottom_right = points["bottom_right"]

    # Width and height of the new transformed image
    width = int(np.linalg.norm(np.array(top_right) - np.array(top_left)))
    height = int(np.linalg.norm(np.array(bottom_left) - np.array(top_left)))

    # Define the destination points (rectangle corners)
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Define the source points as the detected four corners
    src_points = np.float32([top_left, top_right, bottom_left, bottom_right])

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the image to apply the perspective transformation
    transformed_img = cv2.warpPerspective(img, matrix, (width, height))

    return transformed_img, matrix, width, height

def correct_perspective(img, count, position=[995,1140,1530,1661], margin=[400, 400, 400, 400], debug=False):
    """
    Correct the perspective of an image with a skewed rectangle by applying a perspective transform,
    and modify the target area by setting the specified portion to white pixels.
    
    Args:
    img: The full image to process.
    position: Position of target. [top, bottom, left, right]
    margin: Margin around the target.
    
    Returns:
    modified_target: The modified target with the specified portion turned white.
    """
    
    # Step 1: Load the image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((11,11), np.uint8)
    img_opening = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11,11), np.uint8)
    img_closing = cv2.morphologyEx(img_opening, cv2.MORPH_CLOSE, kernel)
    img_clear = img_closing
    if debug:
        cv2.imshow('Clear Image', img_clear)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    target_box = [position[0]-margin[0], position[1]+margin[1],
                position[2]-margin[2], position[3]+margin[3]]
    target = img_clear[target_box[0]:target_box[1], target_box[2]:target_box[3]]
    if debug:
        cv2.imshow('Target', target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite('target.jpg', target)
    
    # Get the dimensions of the target
    h, w = target.shape[:2]
    
    # Define the region you want to turn white
    top_bound = int(0.8 * h)  # 0.8x height from the top
    left_bound = int(0.4 * w)  # 0.5x width from the left
    right_bound = int(0.85 * w)  # 0.8x width from the left (i.e., 0.3x width further)
    
    # Set the specified region to white
    target[top_bound:, left_bound:right_bound] = 255  # 255 corresponds to white in grayscale
    
    # Display the modified target
    if debug:
        cv2.imshow('Modified Target', target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the modified target
    # cv2.imwrite('modified_target.jpg', target)
    
    points = detect_lines_and_points(target, count=count, debug=debug)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_color = img[target_box[0]:target_box[1], target_box[2]:target_box[3]]
    transformed_target, matrix, width, height = apply_perspective_transform(target_color, points)
    if debug:
        cv2.imshow('Transformed_Target', transformed_target)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return transformed_target

if __name__ == '__main__':
    corrected_image = correct_perspective('./experimental_data/test.png')
