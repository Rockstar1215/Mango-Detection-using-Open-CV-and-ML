from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from flask import send_from_directory

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif','webp'}  # Set the allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Add a route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Set the upload folder path in the Flask app configuration
app.config['UPLOAD_FOLDER'] = r'C:\Users\hp\PycharmProjects\cv_ml\uploads'


# Function to check if the upload folder exists, and create it if it doesn't
def ensure_upload_folder_exists():
    UPLOAD_FOLDER = r'C:\Users\hp\PycharmProjects\cv_ml\uploads'
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
        print(f"Upload folder created at: {UPLOAD_FOLDER}")
    else:
        print(f"Upload folder already exists at: {UPLOAD_FOLDER}")


# Call the function to ensure the upload folder exists
ensure_upload_folder_exists()


# Function to perform mango detection
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


# Define a function to load images and labels from a given folder
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels


folder_ripe_test = '/app/Ripe/Test'
folder_ripe_training = '/app/Ripe/Training'
folder_early_ripe_test = '/app/Early_Ripe/Test'
folder_early_ripe_training = '/app/Early_Ripe/Training'
folder_partially_ripe_test = '/app/Partially_Ripe/Test'
folder_partially_ripe_training = '/app/Partially_Ripe/Training'
folder_over_ripe_test = '/app/over_ripe/Test'  # Adjusted the folder name
folder_over_ripe_training = '/app/over_ripe/Training'  # Adjusted the folder name
folder_not_mango_test = '/app/not_mango/Test'  # Adjusted the folder name
folder_not_mango_training = '/app/not_mango/Training'  # Adjusted the folder name
folder_unripe_test='/app/Unripe/Test'
folder_unripe_training='/app/Unripe/Training'

# Load images and labels from each stage
images_ripe_test, labels_ripe_test = load_images_from_folder(folder_ripe_test, 'Ripe')
images_ripe_training, labels_ripe_training = load_images_from_folder(folder_ripe_training, 'Ripe')
images_unripe_test, labels_unripe_test = load_images_from_folder(folder_unripe_test, 'Unripe')
images_unripe_training, labels_unripe_training = load_images_from_folder(folder_unripe_training, 'Unripe')
images_early_ripe_test, labels_early_ripe_test = load_images_from_folder(folder_early_ripe_test, 'Early_Ripe')
images_early_ripe_training, labels_early_ripe_training = load_images_from_folder(folder_early_ripe_training,
                                                                                 'Early_Ripe')
images_partially_ripe_test, labels_partially_ripe_test = load_images_from_folder(folder_partially_ripe_test,
                                                                                 'Partially_Ripe')
images_partially_ripe_training, labels_partially_ripe_training = load_images_from_folder(folder_partially_ripe_training,
                                                                                         'Partially_Ripe')
images_over_ripe_test, labels_over_ripe_test = load_images_from_folder(folder_over_ripe_test, 'over_ripe')
images_over_ripe_training, labels_over_ripe_training = load_images_from_folder(folder_over_ripe_training, 'over_ripe')
images_not_mango_training, labels_not_mango_training = load_images_from_folder(folder_not_mango_training, 'not_mango')
images_not_mango_test, labels_not_mango_test = load_images_from_folder(folder_not_mango_test, 'not_mango')
# Combine images and labels from all stages
images = (images_unripe_test + images_unripe_training +
          images_early_ripe_test + images_early_ripe_training +
          images_partially_ripe_test + images_partially_ripe_training +
          images_ripe_test + images_ripe_training + images_over_ripe_test + images_over_ripe_training + images_not_mango_test +
          images_not_mango_training)

labels = (labels_unripe_test + labels_unripe_training +
          labels_early_ripe_test + labels_early_ripe_training +
          labels_partially_ripe_test + labels_partially_ripe_training +
          labels_ripe_test + labels_ripe_training + labels_over_ripe_test + labels_over_ripe_training +
          labels_not_mango_training + labels_not_mango_test)


# Preprocess images similarly for all images
def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        img = cv2.resize(img, (64, 64))  # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img / 255.0  # Normalize pixel values
        preprocessed_images.append(img)
    return preprocessed_images


preprocessed_images = preprocess_images(images)

X = [img.flatten() for img in preprocessed_images]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.272)

# Define and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)


# Function to perform mango ripeness prediction
def mango_ripeness_prediction(image):
    img = cv2.resize(image, (64, 64))  # Resize image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize pixel values
    img_flattened = img.flatten()
    prediction = mlp.predict([img_flattened])
    return prediction[0]


@app.route('/mango_detection', methods=['POST'])
def detect_mango():
    result = None
    image_path = None
    filename = None  # Define filename variable here

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


# Route handler for mango ripeness prediction
@app.route('/mango_ripeness_prediction', methods=['POST'])
def predict_mango_ripeness():
    result = None

    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            upload_folder = r'C:\Users\hp\PycharmProjects\cv_ml\upload_folder'
            os.makedirs(upload_folder, exist_ok=True)
            image_location = os.path.join(upload_folder, secure_filename(image.filename))
            image.save(image_location)
            img = cv2.imread(image_location)
            if img is not None:
                result = mango_ripeness_prediction(img)
    return jsonify({'result': result})


# Route handler for the root endpoint
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle POST request
        result = None
        image_path = None

        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image_path = 'temp_image.jpg'  # Use the same file name as before
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
                image.save(full_path)
                if os.path.exists(full_path):
                    image = cv2.imread(full_path)
                    if image is not None:
                        result = mango_detection(image)
                    else:
                        result = "Failed to read the uploaded image"
                else:
                    result = "Failed to save the uploaded image"
            else:
                result = "No file uploaded"
        else:
            result = "No image field found in the request"

        return jsonify({'result': result, 'image_path': image_path})
    else:
        # Handle GET request
        # Render the index.html from the 'MANGO' templates folder
        return render_template('index.html')


# Route handler for mango detection page
@app.route('/mango_detection_page')
def mango_detection_page():
    # Render the index_mango_detection.html from the 'Mango_Detection' templates folder
    return render_template('index_mango_detection.html')


# Route handler for mango ripeness prediction page
@app.route('/mango_ripeness_prediction_page')
def mango_ripeness_prediction_page():
    # Render the index_mango_ripeness.html from the 'MANGO' templates folder
    return render_template('index_mango_ripeness.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
