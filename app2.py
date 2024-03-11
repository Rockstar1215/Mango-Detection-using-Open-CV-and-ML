from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)


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


# Define your dataset folders
# Adjusted folder paths inside the Docker container
# Define your dataset folders
# Adjusted folder paths inside the Docker container
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


# Preprocess images
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
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)


# Define and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

# Make predictions on the test set
predictions = mlp.predict(X_test)

# Print classification report and confusion matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=1))

@app.route('/')
def index():
    return render_template('index.html')

# Flask route to accept and process new images
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            upload_folder = r'C:\Users\hp\PycharmProjects\cv_ml\upload_folder'
            os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist
            image_location = os.path.join(upload_folder, secure_filename(image_file.filename))
            image_file.save(image_location)
            img = cv2.imread(image_location)
            img = cv2.resize(img, (64, 64))  # Resize image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = img / 255.0  # Normalize pixel values
            img_flattened = img.flatten()
            prediction = mlp.predict([img_flattened])
            return jsonify({'prediction': prediction[0]})
    return render_template('index_mango_ripeness.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)