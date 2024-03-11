FROM python:3.11

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

ADD combined_app.py .
# Copy mango detection function
COPY app1.py .
COPY Ripe /app/Ripe
COPY Early_Ripe /app/Early_Ripe
COPY Partially_Ripe /app/Partially_Ripe
COPY over_ripe /app/over_ripe
COPY not_mango /app/not_mango
COPY Unripe /app/Unripe
COPY templates /app/templates
# Install dependencies
RUN pip install --upgrade pip
RUN pip install Flask Werkzeug numpy opencv-python scikit-learn joblib python-dotenv 



# Set the command to run both Python scripts sequentially
CMD ["bash", "-c", "python combined_app.py"]
