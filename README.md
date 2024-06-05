# Depression Detection in Children with Voice

This repository contains the source code for a final year project Depression Detection in Children with Voice. The project leverages machine learning models to predict depression in children based on their voice recordings. The application is built using Flask (Python), JavaScript, HTML, CSS, and MySQL.

## Features

- **User Dashboard**: View user details such as name, age, gender, previous prediction results, and trained models. Users can also delete their data from the dashboard.
- **Prediction**: Users can predict depression by uploading a voice file or recording audio. They can use four pre-trained models:
  - Logistic Regression
  - Random Forest
  - SVM Gaussian Kernel
  - SVM Linear Kernel
  - Additionally, users can use their own trained model for prediction.
After making predictions, users can download a detailed prediction report.
- **Model Training**: Users can train their own models by naming their model, uploading normal and stressed voice recordings, and selecting the training algorithm (Logistic Regression, Random Forest, SVM with Gaussian Kernel, or SVM with Linear Kernel). Users can then download the training report or the trained model.

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: MySQL

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamzaaumerr/Depression-Detection-in-Children-with-Voice.git
   cd Depression-Detection-in-Children-with-Voice
   ```

2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. Configure the database:
   - Start Apache and MySQL services from the XAMPP control panel.
   - Open phpMyAdmin by navigating to http://localhost/phpmyadmin/ in your web browser.

4. Run the application:
   ```bash
   flask run
   ```
   or
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:5000`.

## Usage

1. **Sign Up / Log In**: Create an account or log in to the application.
2. **Dashboard**: Access your user dashboard to view and manage your details, previous predictions and model trainings.
3. **Predict Depression**:
   - Upload a voice file or record audio.
   - Choose from the pre-trained models or use your own trained model.
   - Download the prediction report.
4. **Train a Model**:
   - Name your model.
   - Upload normal and stressed voice recordings.
   - Select the training algorithm.
   - Download the training report or the trained model.

## Video Demonstration

https://github.com/hamzaaumerr/Depression-Detection-in-Children-with-Voice/assets/82153059/7a63fe09-c18f-471e-a1e8-23989777c984

**Note**: This project is developed for educational purposes and is part of a final year academic project. The accuracy and reliability of the depression detection models should be validated in a clinical setting before any real-world application.
