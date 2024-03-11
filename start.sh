#!/bin/bash

# Start the Mango_Detection Flask application
python /app/Mango_Detection/app.py &

# Start the Mango-ML67 Flask application
python /app/MANGO-ML67/MANGO-ML/MANGO/app.py
