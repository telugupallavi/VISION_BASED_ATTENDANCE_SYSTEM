import cv2
import sqlite3
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
import random
from tkinter import messagebox, ttk, simpledialog
from datetime import datetime
from PIL import Image, ImageTk

class VisionBasedAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Attendance System - 2026 Optimized")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f5f6fa")

        self.db_path = "vision_attendance.db"
        self.init_database()

        self.cap = None
        self.is_running = False
        
        # Liveness States
        self.current_challenge = None
        self.challenge_met = False
        self.challenges = ["Blink Eyes", "Turn Left", "Turn Right", "Look Up", "Look Down", "Open Mouth", "Smile"]

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.setup_ui()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS users(adm_no TEXT PRIMARY KEY, name TEXT, biometric_vector TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS attendance(adm_no TEXT, name TEXT, date TEXT, time TEXT, status TEXT)")
        conn.commit()
        conn.close()

    def setup_ui(self):
        sidebar = tk.Frame(self.root, bg="#2c3e50", width=280)
        sidebar.pack(side="left", fill="y")

        tk.Label(sidebar, text="CONTROL PANEL", fg="white", bg="#2c3e50", 
                 font=("Segoe UI", 16, "bold")).pack(pady=40)

        self.menu_btn(sidebar, "1. Student Registration", self.register_popup)
        self.menu_btn(sidebar, "2. Manage Users", self.manage_users)
        self.menu_btn(sidebar, "3. Mark Attendance", self.start_attendance_process, color_mode="green")
        self.menu_btn(sidebar, "4. Export Attendance", self.export_popup, color_mode="blue")

        self.display = tk.Frame(self.root, bg="#f5f6fa")
        self.display.pack(side="right", fill="both", expand=True)

        self.instruction_label = tk.Label(self.display, text="System Ready", 
                                         font=("Segoe UI", 20, "bold"), fg="#2c3e50", bg="#f5f6fa")
        self.instruction_label.pack(pady=15)

        self.video_label = tk.Label(self.display, bg="black", width=700, height=450)
        self.video_label.pack(pady=5)

    def menu_btn(self, parent, text, cmd, color_mode="dark"):
        bg, abg = ("#27ae60", "#2ecc71") if color_mode == "green" else (("#3498db", "#5dade2") if color_mode == "blue" else ("#34495e", "#2c3e50"))
        btn = tk.Button(parent, text=text, command=cmd, bg=bg, fg="white", font=("Segoe UI", 10, "bold"), 
                        relief="flat", height=2, width=28, cursor="hand2", activebackground=abg)
        btn.pack(pady=10)
        return btn

    def get_dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def extract_biometric_vector(self, landmarks):
        indices = [33, 133, 362, 263, 1, 61, 291, 199, 10, 152] 
        eye_dist = self.get_dist(landmarks[33], landmarks[263])
        
        points = []
        origin = landmarks[1]
        for i in indices:
            points.append((landmarks[i].x - origin.x) / eye_dist)
            points.append((landmarks[i].y - origin.y) / eye_dist)
        return np.array(points)

    def start_attendance_process(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.challenge_met = False
            self.current_challenge = random.choice(self.challenges)
            self.instruction_label.config(text=f"STEP 1: {self.current_challenge}", fg="#e67e22")
            self.run_recognition_loop()

    def run_recognition_loop(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret: self.stop_camera(); return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            self.mp_draw.draw_landmarks(frame, result.multi_face_landmarks[0], self.mp_face_mesh.FACEMESH_CONTOURS)
            
            if not self.challenge_met:
                if self.verify_challenge(face):
                    self.challenge_met = True
                    self.instruction_label.config(text="STEP 2: Look Straight at Camera", fg="#2980b9")
            else:
                nose, l_eye, r_eye = face[1], face[33], face[263]
                alignment_score = abs((nose.x - l_eye.x) - (r_eye.x - nose.x))
                if alignment_score < 0.04:
                    vector = self.extract_biometric_vector(face)
                    if self.identify_and_save(vector):
                        self.stop_camera(); return
                else:
                    self.instruction_label.config(text="Please Center Your Face", fg="#d35400")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.root.after(10, self.run_recognition_loop)

    def verify_challenge(self, face):
        ear = (self.get_dist(face[159], face[145]) + self.get_dist(face[386], face[374])) / 2.0
        m_gap = self.get_dist(face[13], face[14])
        s_gap = self.get_dist(face[61], face[291])
        n, f, c, le, re = face[1], face[10], face[152], face[33], face[263]

        if self.current_challenge == "Blink Eyes" and ear < 0.02: return True
        if self.current_challenge == "Open Mouth" and m_gap > 0.05: return True
        if self.current_challenge == "Smile" and s_gap > 0.1: return True
        if self.current_challenge == "Turn Left" and n.x < le.x: return True
        if self.current_challenge == "Turn Right" and n.x > re.x: return True
        if self.current_challenge == "Look Up" and (n.y - f.y) < 0.14: return True
        if self.current_challenge == "Look Down" and (c.y - n.y) < 0.14: return True
        return False

    def identify_and_save(self, current_vector):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT adm_no, name, biometric_vector FROM users")
        rows = cursor.fetchall()
        
        best_match = None
        min_dist = 0.28  # TIGHTENED THRESHOLD: Lower values = stricter matching

        for adm, name, data_str in rows:
            stored_vector = np.array(list(map(float, data_str.split(","))))
            dist = np.linalg.norm(current_vector - stored_vector)
            
            if dist < min_dist:
                min_dist = dist
                best_match = (adm, name)

        if best_match:
            adm, name = best_match
            now = datetime.now()
            today = str(now.date())
            cursor.execute("SELECT * FROM attendance WHERE adm_no=? AND date=?", (adm, today))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO attendance VALUES (?,?,?,?,?)",
                              (adm, name, today, now.strftime("%H:%M:%S"), "Present"))
                conn.commit()
                messagebox.showinfo("Success", f"Attendance Marked: {name}")
            else:
                messagebox.showwarning("Notice", f"Attendance for {name} has already been marked today.")
            conn.close()
            return True
        else:
            # If no one in the DB is close enough to the face in the camera
            messagebox.showerror("Not Recognized", "User not found. Please register before marking attendance.")
            conn.close()
            return True # Stops camera to prevent spamming errors

    def stop_camera(self):
        self.is_running = False
        if self.cap: self.cap.release()
        self.video_label.config(image="")
        self.instruction_label.config(text="System Ready", fg="#2c3e50")

    def register_popup(self):
        pop = tk.Toplevel(self.root); pop.geometry("300x350"); pop.title("Register")
        tk.Label(pop, text="Full Name").pack(pady=5); n_ent = tk.Entry(pop); n_ent.pack()
        tk.Label(pop, text="Admission ID").pack(pady=5); a_ent = tk.Entry(pop); a_ent.pack()

        def capture():
            if not n_ent.get() or not a_ent.get(): return
            c = cv2.VideoCapture(0); ret, frame = c.read(); c.release()
            if ret:
                res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    v = self.extract_biometric_vector(res.multi_face_landmarks[0].landmark)
                    conn = sqlite3.connect(self.db_path)
                    try:
                        conn.execute("INSERT INTO users VALUES (?,?,?)", (a_ent.get(), n_ent.get(), ",".join(map(str, v))))
                        conn.commit(); messagebox.showinfo("Saved", f"Registered {n_ent.get()}!"); pop.destroy()
                    except: messagebox.showerror("Error", "ID already exists!")
                    conn.close()
                else: messagebox.showwarning("Error", "No face detected. Look at the camera.")

        tk.Button(pop, text="Capture & Save", command=capture, bg="#27ae60", fg="white", height=2).pack(pady=20)

    def manage_users(self):
        win = tk.Toplevel(self.root)
        win.title("Manage Student Records")
        win.geometry("600x500")
        
        tree = ttk.Treeview(win, columns=("ID", "Name"), show="headings")
        tree.heading("ID", text="Adm No")
        tree.heading("Name", text="Name")
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        def refresh_table():
            for item in tree.get_children(): tree.delete(item)
            conn = sqlite3.connect(self.db_path)
            for r in conn.execute("SELECT adm_no, name FROM users"):
                tree.insert("", "end", values=r)
            conn.close()

        refresh_table()

        btn_frame = tk.Frame(win)
        btn_frame.pack(pady=10)

        def handle_edit():
            selected = tree.selection()
            if not selected: return
            item = tree.item(selected[0])['values']
            self.edit_user(item[0], item[1], refresh_table)

        def handle_delete():
            selected = tree.selection()
            if not selected: return
            item = tree.item(selected[0])['values']
            self.delete_user(item[0], refresh_table)

        tk.Button(btn_frame, text="Edit Student", command=handle_edit, bg="#3498db", fg="white", width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Delete Student", command=handle_delete, bg="#e74c3c", fg="white", width=15).pack(side="left", padx=5)

    def edit_user(self, old_id, old_name, callback):
        new_name = simpledialog.askstring("Edit User", "Enter new name:", initialvalue=old_name)
        if not new_name: return
        new_id = simpledialog.askstring("Edit User", "Enter new Admission ID:", initialvalue=old_id)
        if not new_id: return

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE users SET name=?, adm_no=? WHERE adm_no=?", (new_name, new_id, old_id))
            conn.execute("UPDATE attendance SET name=?, adm_no=? WHERE adm_no=?", (new_name, new_id, old_id))
            conn.commit()
            messagebox.showinfo("Success", "Record Updated!")
            callback()
        except Exception as e:
            messagebox.showerror("Error", f"Could not update: {e}")
        finally:
            conn.close()

    def delete_user(self, adm_no, callback):
        if messagebox.askyesno("Confirm Delete", f"Delete student {adm_no} and all attendance records?"):
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM users WHERE adm_no=?", (adm_no,))
            conn.execute("DELETE FROM attendance WHERE adm_no=?", (adm_no,))
            conn.commit()
            conn.close()
            callback()

    def export_popup(self):
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql("SELECT * FROM attendance", conn)
            df.to_excel("Attendance_Report_2026.xlsx", index=False)
            conn.close()
            messagebox.showinfo("Export", "Report saved as Attendance_Report_2026.xlsx")
        except Exception as e: messagebox.showerror("Export Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VisionBasedAttendanceSystem(root)
    root.mainloop()