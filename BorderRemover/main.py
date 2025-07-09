import os
import cv2

INPUT = 'input'
OUTPUT = 'output'

def remove_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_dilate = cv2.dilate(edges, None, iterations=3)
    contours, _ = cv2.findContours(edges_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    cropped = img[y:y+h, x:x+w]
    return cropped

def main():
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    for filename in os.listdir(INPUT):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(INPUT, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: {filename}")
                continue

            cropped = remove_border(img)
            out_path = os.path.join(OUTPUT, filename)
            cv2.imwrite(out_path, cropped)

    print("Processed images are saved in output file.")

if __name__ == "__main__":
    main()

