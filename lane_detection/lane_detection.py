import cv2
import numpy as np

# === Reuse These Functions ===

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(100, height), (img.shape[1]-100, height), (img.shape[1]//2, height//2 + 50)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def make_coordinates(image, slope, intercept):
    height = image.shape[0]
    y1 = height
    y2 = int(height * 0.6)
    if slope == 0:
        slope = 0.1
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    averaged_lines = []
    if left_lines:
        left_avg = np.mean(left_lines, axis=0)
        averaged_lines.append(make_coordinates(image, *left_avg))

    if right_lines:
        right_avg = np.mean(right_lines, axis=0)
        averaged_lines.append(make_coordinates(image, *right_avg))

    return averaged_lines

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    averaged_lines = average_slope_intercept(image, lines)
    for line in averaged_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# === Main Processing ===

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=150)
    line_img = display_lines(frame, lines)
    combo = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combo

def main():
    cap = cv2.VideoCapture("res/test_video.mp4")

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lane_frame = process_frame(frame)
        cv2.imshow("Lane Detection Video", lane_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
