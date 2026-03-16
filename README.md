# Vision Based Contactless Attendance System Using Facial Mesh and Iris Landmark Analysis (2026 Optimized)

This project implements a Vision Based Attendance System using Face Recognition and Liveness Detection.  
It automatically identifies registered students and marks their attendance.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Tkinter GUI
- SQLite Database

## Features
- Student Registration with Face Capture
- Face Recognition Attendance
- Liveness Detection (Blink, Smile, Head Movement)
- User Management (Edit/Delete Students)
- Attendance Storage in SQLite Database
- Export Attendance Report to Excel
- GUI Interface using Tkinter

## System Workflow
1. Register student using camera capture
2. Store biometric vector in database
3. Perform liveness challenge (blink, smile, head movement)
4. Recognize face using biometric vector comparison
5. Mark attendance automatically
6. Export attendance to Excel file

## Installation

Install required libraries:

pip install -r requirements.txt

## Run the Project

Run the main Python file:

python vision_attendance.py

## Project Structure

vision-attendance-system
│
├── vision_attendance.py
├── requirements.txt
├── README.md

## Output

- Attendance stored in SQLite database
- Exported report: Attendance_Report_2026.xlsx

## Author
Telugu Pallavi
Vision Based Attendance System
