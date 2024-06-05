import os
import bcrypt
import init_db
import load_lr
import load_rf
import load_svmlk
import load_svmgk
from fpdf import FPDF
import mysql.connector
import preprocess as pre
import RandomForest as RF
from flask import send_file
from dotenv import load_dotenv
import LogisticRegression as LR
import SVMLinearKernel as SVMLK
import SVMGaussianKernel as SVMGK
from flask_socketio import SocketIO
from flask_mail import Mail, Message
import extract_feature_Normal as EFNormal
from werkzeug.utils import secure_filename
import extract_feature_Stressed as EFStressed
from flask import Flask, redirect, render_template, request, session, url_for, flash

import recordingaudio as reco

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "default_secret_key")
socketio = SocketIO(app)

# Load environment variables from .env file
load_dotenv()

# Configure mail settings using environment variables
app.config["MAIL_SERVER"] = os.environ.get("MAIL_SERVER")
app.config["MAIL_PORT"] = os.environ.get("MAIL_PORT")
app.config["MAIL_USE_TLS"] = os.environ.get("MAIL_USE_TLS")
app.config["MAIL_EMAIL"] = os.environ.get("MAIL_EMAIL")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")

# Instantiate the mail object
mail = Mail(app)

# Connect to MySQL database using environment variables
db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    port=os.getenv("MYSQL_PORT"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)

# cursor = db.cursor()

# Define the upload and recording folder
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

RECORDINGS_FOLDER = os.getenv("RECORDINGS_FOLDER")
app.config["RECORDINGS_FOLDER"] = RECORDINGS_FOLDER

MODELS_FOLDER = os.getenv("MODELS_FOLDER")
app.config["MODELS_FOLDER"] = MODELS_FOLDER

# Define allowed extensions for file uploads
ALLOWED_EXTENSIONS = {"wav"}


# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# working
@app.route("/")
def index():
    if not session.get("logged_in"):
        return render_template("index.html")
    else:
        # Get the user ID from the session
        user_id = session.get("user_id")
        cursor = db.cursor()
        # Fetch data from the predict table for the current user
        predict_query = "SELECT id, filename, model, prediction_class, prediction_percentage FROM predict WHERE patient_id = %s"
        cursor.execute(predict_query, (user_id,))
        predict_data = cursor.fetchall()

        # Fetch data from the model table for the current user
        model_query = "SELECT id, model_name, model_choice, mse, mae, r2, rms, accuracy FROM model WHERE patient_id = %s"
        cursor.execute(model_query, (user_id,))
        model_data = cursor.fetchall()

        # Close the cursor and database connection
        cursor.close()

        return render_template(
            "dashboard.html", predict_data=predict_data, model_data=model_data
        )


# Sign up
@app.route("/signup", methods=["GET", "POST"])
def sign_up_page():
    if request.method == "GET":
        return render_template("sign-up.html")
    elif request.method == "POST":
        # Get form data
        firstname = request.form["firstname"]
        lastname = request.form["lastname"]
        gender = request.form["gender"]
        age = request.form["age"]
        email = request.form["email"]
        password = request.form["password"]

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        cursor = db.cursor()
        # Insert user data into the database
        cursor.execute(
            "INSERT INTO patient (password, email, first_name, last_name, age, gender) VALUES (%s, %s, %s, %s, %s, %s)",
            (hashed_password, email, firstname, lastname, age, gender),
        )
        db.commit()
        cursor.close()
        return render_template("sign-in.html")
    pass


# Sign in
@app.route("/signin", methods=["GET", "POST"])
def sign_in_page():
    if request.method == "GET":
        return render_template("sign-in.html")
    elif request.method == "POST":
        # Get form data
        email = request.form["email"]
        password = request.form["password"]
        cursor = db.cursor()
        # Query the database for the user
        cursor.execute("SELECT * FROM patient WHERE email = %s", (email,))
        user = cursor.fetchone()

        # Check if the user exists and the password is correct
        if user and bcrypt.checkpw(password.encode("utf-8"), user[1].encode("utf-8")):
            # Store user data in the session
            session["logged_in"] = True
            session["user_id"] = user[0]
            session["email"] = user[2]
            session["first_name"] = user[3]
            session["last_name"] = user[4]
            session["age"] = user[5]
            session["gender"] = user[6]

            # Get the user ID from the session
            user_id = session.get("user_id")

            # Fetch data from the predict table for the current user
            predict_query = "SELECT id, filename, model, prediction_class, prediction_percentage FROM predict WHERE patient_id = %s"
            cursor.execute(predict_query, (user_id,))
            predict_data = cursor.fetchall()

            # Fetch data from the model table for the current user
            model_query = "SELECT id, model_name, model_choice, mse, mae, r2, rms, accuracy FROM model WHERE patient_id = %s"
            cursor.execute(model_query, (user_id,))
            model_data = cursor.fetchall()
            cursor.close()

            return render_template(
                "dashboard.html", predict_data=predict_data, model_data=model_data
            )
        else:
            cursor.close()
            # Invalid login credentials
            return render_template("sign-in.html")
    pass


# Route to delete a prediction
@app.route("/delete_prediction/<int:prediction_id>", methods=["POST"])
def delete_prediction(prediction_id):
    # Get the user ID from the session
    user_id = session.get("user_id")
    cursor = db.cursor()
    # Execute the SQL query to delete the prediction with the given ID and user_id
    delete_query = "DELETE FROM predict WHERE id = %s AND patient_id = %s"
    cursor.execute(delete_query, (prediction_id, user_id))
    db.commit()
    cursor.close()
    return redirect(url_for("index"))


# Route to delete a model
@app.route("/delete_model/<int:model_id>", methods=["POST"])
def delete_model(model_id):
    # Get the user ID from the session
    user_id = session.get("user_id")
    cursor = db.cursor()
    # Execute the SQL query to delete the model with the given ID and user_id
    delete_query = "DELETE FROM model WHERE id = %s AND patient_id = %s"
    cursor.execute(delete_query, (model_id, user_id))
    db.commit()
    cursor.close()
    return redirect(url_for("index"))


# Route to delete all records from the 'model' table for the logged-in user
@app.route("/delete_all_models", methods=["POST"])
def delete_all_models():
    # Get the user ID from the session
    user_id = session.get("user_id")
    cursor = db.cursor()
    # Execute the SQL query to delete all records from the 'model' table for the logged-in user
    delete_query = "DELETE FROM model WHERE patient_id = %s"
    cursor.execute(delete_query, (user_id,))
    db.commit()
    cursor.close()
    return redirect(url_for("index"))


# Route to delete all records from the 'predict' table for the logged-in user
@app.route("/delete_all_predictions", methods=["POST"])
def delete_all_predictions():
    # Get the user ID from the session
    user_id = session.get("user_id")
    cursor = db.cursor()
    # Execute the SQL query to delete all records from the 'predict' table for the logged-in user
    delete_query = "DELETE FROM predict WHERE patient_id = %s"
    cursor.execute(delete_query, (user_id,))
    db.commit()
    cursor.close()
    return redirect(url_for("index"))


# working
@app.route("/signout", methods=["GET"])
def sign_out():
    # Clear the session
    session.clear()
    return render_template("index.html")


# working
@app.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html")


# working
@app.route("/contact", methods=["GET", "POST"])
def contact_page():
    if request.method == "GET":
        return render_template("contact.html")
    elif request.method == "POST":
        # Get form data
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        # Send email
        msg = Message(subject, sender=email, recipients=["hamzakhann0666@gmail.com"])
        msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        mail.send(msg)

        return "Email sent successfully!"

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "GET":
        # Get the user ID from the session
        user_id = session.get("user_id")
        cursor = db.cursor()
        # Fetch data from the model table for the current user
        model_query = "SELECT model_name, model_choice, model_path FROM model WHERE patient_id = %s"
        cursor.execute(model_query, (user_id,))
        model_names = cursor.fetchall()
        cursor.close()

        return render_template("predict.html", model_names=model_names)
    elif request.method == "POST":
        # Get the form data
        selected_model = request.form["model"]

        # Check if the user chose to record voice or upload voice
        if request.form.get("voiceOption") == "record":
            file_name = request.form["voiceName"] + ".wav"
            # Process the recorded voice
            recording_path = os.path.join(app.config["RECORDINGS_FOLDER"], file_name)
            audio_file = recording_path
        else:
            # Get the uploaded audio file
            audio_file = request.files["voiceFile"]

            # Check if a file is uploaded
            if audio_file.filename == "":
                return "No file selected. Please upload an audio file."

            # Ensure the file is a WAV file
            if not audio_file.filename.endswith(".wav"):
                return "Invalid file format. Please upload a .wav file."
            
            file_name = audio_file.filename

        # Get the user ID from the session
        user_id = session.get("user_id")
        cursor = db.cursor()
        # Fetch data from the model table for the current user
        model_query = "SELECT model_name, model_choice, model_path FROM model WHERE patient_id = %s"
        cursor.execute(model_query, (user_id,))
        model_names = cursor.fetchall()

        # Find the model_path for the selected_model
        model_path = None

        # Extract model name and path from selected_model value
        selected_model_parts = selected_model.split(maxsplit=1)
        selected_model_name = selected_model_parts[0]
        print(selected_model_name)
        selected_model_path = selected_model_parts[1]
        print(selected_model_path)
        # Use the extracted model path
        model_path = selected_model_path

        # Perform prediction based on the selected model
        if selected_model_name == "logistic_regression":
            prediction_result = load_lr.process(audio_file, model_path)
        elif selected_model_name == "random_forest":
            prediction_result = load_rf.process(audio_file, model_path)
        elif selected_model_name == "svm_gaussian_kernel":
            prediction_result = load_svmgk.process(audio_file, model_path)
        elif selected_model_name == "svm_linear_kernel":
            prediction_result = load_svmlk.process(audio_file, model_path)
        else:
            return "Invalid model selection."

        # Extract prediction class and percentage
        prediction_class = prediction_result.split(":")[0].strip()
        prediction_percentage = prediction_result.split(":")[1].strip()

        # Save the results to the database
        cursor.execute(
            "INSERT INTO predict (patient_id, filename, model, prediction_class, prediction_percentage) VALUES (%s, %s, %s, %s, %s)",
            (
                user_id,
                file_name,
                selected_model_name,
                prediction_class,
                prediction_percentage,
            ),
        )
        db.commit()
        cursor.close()

        # Render a template with the prediction result
        return render_template(
            "predict.html",
            file_name=file_name,
            selected_model=selected_model_name,
            prediction_class=prediction_class,
            prediction_percentage=prediction_percentage,
            model_names=model_names,
        )


@app.route("/download_prediction_report", methods=["GET"])
def download_predict_report():
    # Get data from the session
    file_name = request.args.get("file_name")
    prediction_class = request.args.get("prediction_class")
    prediction_percentage = request.args.get("prediction_percentage")
    selected_model = request.args.get("selected_model")
    first_name = session.get("first_name")
    last_name = session.get("last_name")
    age = session.get("age")
    gender = session.get("gender")

    # Create a PDF document
    pdf_file_path = "prediction_report.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Set up font
    pdf.set_font("Arial", size=12)

    # Write content to PDF
    pdf.cell(200, 10, txt="Prediction Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Name: {first_name} {last_name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
    pdf.cell(200, 10, txt=f"File Name: {file_name}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {prediction_class}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction Percentage: {prediction_percentage}", ln=True)
    pdf.cell(200, 10, txt=f"Model Used: {selected_model}", ln=True)

    # Save PDF
    pdf.output(pdf_file_path)

    # Send the PDF file as a downloadable attachment
    return send_file(pdf_file_path, as_attachment=True)


# not working
@app.route("/recording", methods=["POST"])
def recording():
    name = request.form["voiceName"]
    # Start recording process
    recording_path = os.path.join(app.config["RECORDINGS_FOLDER"], name + ".wav")
    reco.process(recording_path)
    # Return success response
    return "Recording started successfully."


# working
@app.route("/trainmodel", methods=["GET", "POST"])
def train_model_page():
    if request.method == "GET":
        return render_template("train-model.html")
    elif request.method == "POST":
        # Get the user ID from the session
        user_id = session.get("user_id")

        # Check if files were uploaded
        if "normalFile" not in request.files or "stressedFile" not in request.files:
            return "No files uploaded"

        # Get uploaded files and model choice from the form
        normal_files = request.files.getlist("normalFile")
        stressed_files = request.files.getlist("stressedFile")
        model_choice = request.form["model"]
        model_name = request.form["modelName"]
        file_name = model_name + " " + model_choice + ".joblib"
        model_path = os.path.join(app.config["MODELS_FOLDER"], file_name)

        # Define the directory paths for normal and stressed files inside the dataset folder
        dataset_dir = os.path.join(app.config["UPLOAD_FOLDER"], "dataset")
        normal_dir = os.path.join(dataset_dir, "normal")
        stressed_dir = os.path.join(dataset_dir, "stressed")
        results_dir = "./results"
        dataset_path = os.path.join(results_dir, "dataset.csv")

        # Create the directories if they don't exist
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(stressed_dir, exist_ok=True)

        # Remove previous files from the folders
        for file in os.listdir(normal_dir):
            file_path = os.path.join(normal_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for file in os.listdir(stressed_dir):
            file_path = os.path.join(stressed_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the uploaded files to the respective folders
        for normal_file in normal_files:
            if allowed_file(normal_file.filename):
                normal_file_path = os.path.join(
                    normal_dir, secure_filename(normal_file.filename)
                )
                normal_file.save(normal_file_path)

        for stressed_file in stressed_files:
            if allowed_file(stressed_file.filename):
                stressed_file_path = os.path.join(
                    stressed_dir, secure_filename(stressed_file.filename)
                )
                stressed_file.save(stressed_file_path)

        # Preprocess the data and extract features
        EFNormal.process(normal_dir)
        EFStressed.process(stressed_dir)
        pre.process()

        # Train selected model
        if model_choice == "logistic_regression":
            mse, mae, r2, rms, ac = LR.process(dataset_path, app, model_path)
        elif model_choice == "random_forest":
            mse, mae, r2, rms, ac = RF.process(dataset_path, app, model_path)
        elif model_choice == "svm_gaussian_kernel":
            mse, mae, r2, rms, ac = SVMGK.process(dataset_path, app, model_path)
        elif model_choice == "svm_linear_kernel":
            mse, mae, r2, rms, ac = SVMLK.process(dataset_path, app, model_path)
        else:
            return "Invalid model choice"
        cursor = db.cursor()
        # Save the results to the database
        cursor.execute(
            "INSERT INTO model (patient_id, model_name, model_path, model_choice, mse, mae, r2, rms, accuracy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (user_id, model_name, model_path, model_choice, mse, mae, r2, rms, ac),
        )
        db.commit()
        cursor.close()
        # Return result or redirect to a new route for displaying result
        return render_template(
            "train-model.html",
            model_name = model_name,
            model_choice=model_choice,
            model_mse=mse,
            model_mae=mae,
            model_rsquare=r2,
            model_rmse=rms,
            model_accuracy=ac,
        )
    pass


@app.route('/download_model_file')
def download_model_file():
    # Get the current user's ID from the session
    user_id = session.get("user_id")
    model_name = request.args.get("model_name")
    # model_choice = request.args.get("model_choice")
    cursor = db.cursor()
    # Query the database for the model file path
    cursor.execute("SELECT model_path FROM model WHERE patient_id = %s AND model_name = %s", (user_id, model_name))
    result = cursor.fetchone()
    
    if result:
        model_path = result[0]
        cursor.fetchall()
        cursor.close()
        return send_file(model_path, as_attachment=True)
    else:
        cursor.close()
        return "Model file not found", 404


@app.route("/download_model_report", methods=["GET"])
def download_model_report():
    model_name = request.args.get("model_name")
    model_choice = request.args.get("model_choice")
    model_accuracy = request.args.get("model_accuracy")
    model_mse = request.args.get("model_mse")
    model_rmse = request.args.get("model_rmse")
    model_mae = request.args.get("model_mae")
    model_rsquare = request.args.get("model_rsquare")

    # Create a PDF document
    pdf_file_path = "model_training_report.pdf"
    pdf = FPDF()
    pdf.add_page()

    # Set up font
    pdf.set_font("Arial", size=12)

    # Write content to PDF
    pdf.cell(200, 10, txt="Model Training Report", ln=True, align="C")
    pdf.cell(200, 10, txt=f"Model Name: {model_name}", ln=True)
    pdf.cell(200, 10, txt=f"Selected Model: {model_choice}", ln=True)
    pdf.cell(200, 10, txt=f"Model Accuracy: {model_accuracy}%", ln=True)
    pdf.cell(200, 10, txt=f"Mean Squared Error: {model_mse}%", ln=True)
    pdf.cell(200, 10, txt=f"Root Mean Squared Error: {model_rmse}%", ln=True)
    pdf.cell(200, 10, txt=f"Mean Absolute Error: {model_mae}%", ln=True)
    pdf.cell(200, 10, txt=f"R-Squared: {model_rsquare}%", ln=True)

    # Save PDF
    pdf.output(pdf_file_path)

    # Send the PDF file as a downloadable attachment
    return send_file(pdf_file_path, as_attachment=True)


@socketio.on_error()
def error_handler(e):
    pass


if __name__ == "__main__":
    init_db.init_db()
    socketio.run(app, debug=True, host="127.0.0.1", port=4000)
