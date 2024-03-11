import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory

app = Flask(__name__)
# Set the path to the upload directory
UPLOAD_FOLDER = r'C:\Users\hp\PycharmProjects\cv_ml\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Set the directory where uploaded files will be saved
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif', 'webp'}  # Set the allowed file extensions


# Function to check if the upload folder exists, and create it if it doesn't
def ensure_upload_folder_exists():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Upload folder created at: {UPLOAD_FOLDER}")
    else:
        print(f"Upload folder already exists at: {UPLOAD_FOLDER}")


# Call the function to ensure the upload folder exists
ensure_upload_folder_exists()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def mango_detection(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the color range that represents a mango (adjust these values as needed)
    lower_mango_color = np.array([35, 55, 55])  # Example values for orange color
    upper_mango_color = np.array([65, 260, 260])  # Example values for orange color

    # Create a mask to extract the mango color
    color_mask = cv2.inRange(hsv, lower_mango_color, upper_mango_color)

    # Calculate the percentage of the image covered by mango color
    percentage_mango_color = (np.sum(color_mask) / 255) / (image.shape[0] * image.shape[1]) * 100

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize shape-related features
    aspect_ratio = 0
    contour_area = 0
    contour_solidity = 0
    contour_extent = 0

    # Calculate shape features if contours are found
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h
        contour_area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        contour_solidity = contour_area / cv2.contourArea(hull)
        contour_extent = contour_area / (w * h)

    # Define thresholds for considering it a mango based on color and shape
    mango_color_threshold = 3.1  # Adjust based on color coverage
    mango_aspect_ratio_threshold = 0.9  # Adjust based on shape
    mango_solidity_threshold = 1  # Adjust based on solidity
    mango_extent_threshold = 0.7  # Adjust based on extent

    # Combine color and shape information to classify the object
    if (
            percentage_mango_color >= mango_color_threshold
            and aspect_ratio >= mango_aspect_ratio_threshold
            and contour_solidity >= mango_solidity_threshold
            and contour_extent >= mango_extent_threshold
    ):
        result = "It's a mango"
    else:
        result = "Not a mango"

    return result


@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/', methods=['GET', 'POST'])
def detect_mango():
    result = None
    image_path = None

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                # Set the relative path to the saved image file within UPLOAD_FOLDER
                image_path = 'temp_image.jpg'  # Use the same file name as before

                try:
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
                    image.save(full_path)
                    print(f"Image saved to: {full_path}")  # Print the full path where the image is saved
                    if os.path.exists(full_path):
                        print("Image exists in uploads folder.")
                    else:
                        print("Image does not exist in uploads folder.")
                    image = cv2.imread(full_path)
                    if image is not None:
                        result = mango_detection(image)
                except Exception as e:
                    print(f"Error saving image: {e}")

    return render_template('index_mango_detection.html', result=result, image_path=image_path)


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
