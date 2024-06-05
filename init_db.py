import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def init_db():
    try:
        # Connect to MySQL database using environment variables
        with mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            port=os.getenv("MYSQL_PORT"),
            password=os.getenv("MYSQL_PASSWORD"),
        ) as db:

            # Create the database if it doesn't exist
            with db.cursor() as cursor:
                cursor.execute(
                    "CREATE DATABASE IF NOT EXISTS `{}`".format(os.getenv("MYSQL_DATABASE"))
                )

            # Switch to the newly created or existing database
            db.database = os.getenv("MYSQL_DATABASE")

            # Create tables if they don't exist
            with db.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patient (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        password VARCHAR(255) NOT NULL,
                        email VARCHAR(255),
                        first_name VARCHAR(255),
                        last_name VARCHAR(255),
                        age INT,
                        gender VARCHAR(10)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predict (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        patient_id INT,
                        filename VARCHAR(255),
                        model VARCHAR(50),
                        prediction_class VARCHAR(50),
                        prediction_percentage FLOAT,
                        FOREIGN KEY (patient_id) REFERENCES patient(id)
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        patient_id INT,
                        model_name VARCHAR(255),
                        model_path VARCHAR(255),
                        model_choice VARCHAR(50),
                        mse FLOAT,
                        mae FLOAT,
                        r2 FLOAT,
                        rms FLOAT,
                        accuracy FLOAT,
                        FOREIGN KEY (patient_id) REFERENCES patient(id)
                    )
                    """
                )

            # Commit changes
            db.commit()

    except mysql.connector.Error as err:
        print("Error:", err)

# if __name__ == "__main__":
init_db()