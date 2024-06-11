import os 
from os import environ
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Parsing .pts files
def parse_pts_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines:
            if '{' in line or '}' in line or 'version:' in line or 'n_points:' in line:
                continue
            x, y = map(float, line.strip().split())
            points.append((x, y))
        return points

def insert_landmarks_into_db(cursor, p_id, image_id, points): 
    for index, (x, y) in enumerate(points):
        # Check if the record already exists
        cursor.execute(
            "SELECT COUNT(*) FROM facial_landmarks WHERE p_id = %s AND image_id = %s AND point_id = %s",
            (p_id, image_id, index)
        )
        result = cursor.fetchone()
        if result[0] == 0:  # If record does not exist, insert it
            cursor.execute(
                "INSERT INTO facial_landmarks (p_id, image_id, point_id, x_coordinate, y_coordinate) VALUES (%s, %s, %s, %s, %s)",
                (p_id, image_id, index, x, y)
            )

def main():
    # Load .env file
    load_dotenv()
    
    try:
        # Database connection
        conn = mysql.connector.connect(
            database=environ.get('MYSQL_DB'),
            user=environ.get('MYSQL_USER'),
            password=environ.get('MYSQL_PASSWORD'),
            host=environ.get('MYSQL_HOST'),
            port=environ.get('MYSQL_PORT')
        )

        if conn.is_connected():
            print("Successfully connected to the database")

        cursor = conn.cursor()

        # Directory containing pts files
        script_directory = os.path.dirname(os.path.abspath(__file__))
        pts_files_directory = os.path.join(script_directory, 'Face Markup AR Database/points_22')

        for p_id in os.listdir(pts_files_directory):
            sub_directory_path = os.path.join(pts_files_directory, p_id)
            if os.path.isdir(sub_directory_path):
                for file_name in os.listdir(sub_directory_path):
                    if file_name.endswith('.pts'):
                        file_path = os.path.join(sub_directory_path, file_name)
                        points = parse_pts_file(file_path)
                        image_id = os.path.splitext(file_name)[0]  # Remove the .pts extension
                        insert_landmarks_into_db(cursor, p_id, image_id, points)

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
