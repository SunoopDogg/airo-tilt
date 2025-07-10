import os
import cv2
import numpy as np

ADDRESS = "http://192.168.0.53:4747/video"
# export DISPLAY=192.168.0.58:0 # for MacOS

FILENAME = "IMG_0940.jpg"


def get_tilt(img: cv2.Mat):
    """
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Sharpening filter
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(sharpened, (11, 11), 0)
    cv2.imwrite(os.path.join(os.getcwd(), "output", "blurred.jpg"), blurred)

    # Edge detection
    edges = cv2.Canny(blurred, 20, 60)
    cv2.imwrite(os.path.join(os.getcwd(), "output", "edges.jpg"), edges)

    # Find rectangles in the image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for c in contours:
    #     cv2.drawContours(img, [c], -1, (255, 0, 0), 5)
    # cv2.imwrite(os.path.join(os.getcwd(), "output", "contours.jpg"), img)

    rectangles = [cv2.minAreaRect(c) for c in contours if len(c) >= 4]
    rectangles = [r for r in rectangles if r[1][0] > 100 and r[1][1] > 100]
    rectangles = sorted(
        rectangles, key=lambda r: r[1][0] * r[1][1], reverse=True)

    for rect in rectangles:
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        # Draw rectangles on the image
        cv2.drawContours(img, [box], 0, (0, 255, 0), 5)
        # Draw the center of the rectangle
        center = tuple(map(int, rect[0]))
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        # Calculate the angle of the rectangle
        angle = rect[2]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        cv2.putText(img, f"Angle: {angle:.2f}", (int(center[0]), int(center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(os.getcwd(), "output", "rectangles.jpg"), img)


def real_time():
    # Droid camera
    cap = cv2.VideoCapture(ADDRESS)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        get_tilt(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # Load image
    img = cv2.imread(os.path.join(os.getcwd(), "images", FILENAME))
    if img is None:
        print(f"Error: Could not read image {FILENAME}.")
        exit()

    # active area
    TOP_LEFT = (1000, 750)
    BOTTOM_RIGHT = (3300, 4700)
    img = img[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]]

    get_tilt(img)

    # Show the image
    # cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
