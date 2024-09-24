import os
import cv2
import matplotlib.pyplot as plt

# รายชื่อโฟลเดอร์และตำแหน่งในบริษัท (ตามข้อมูลใหม่)
members = [
    ('Muay', 'P1_CEO'),
    ('Fahfon', 'P2_UXUI'),
    ('Nine', 'P3_Front1'),
    ('Mhooyong', 'P4_Front2'),
    ('First', 'P5_back1'),
    ('Jay', 'P6_back2'),
    ('Golf', 'P7_se'),
    ('Boss', 'P8_sa')
]

# โหลดไฟล์ cascade classifier สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# สร้างโฟลเดอร์ตามรายชื่อที่กำหนด
def create_folders():
    base_path = './group_members/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    for member_name, folder_name in members:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f'Created folder: {folder_path}')
        else:
            print(f'Folder {folder_path} already exists!')

# ฟังก์ชันบันทึกรูปภาพลงในโฟลเดอร์ที่ถูกต้อง ถ้าไฟล์ยังไม่มี
def save_image_to_folders(frame, frame_count, member_idx):
    base_path = './group_members/'
    member_name, folder_name = members[member_idx]
    folder_path = os.path.join(base_path, folder_name)

    # ปรับชื่อไฟล์ให้ตรงกับรูปแบบที่ต้องการ
    image_file_name = f"{folder_name}_{member_name}_{frame_count}.png"
    image_file_path = os.path.join(folder_path, image_file_name)

    # เช็คว่าไฟล์นี้มีอยู่แล้วหรือไม่
    if not os.path.exists(image_file_path):
        cv2.imwrite(image_file_path, frame)
        print(f"Saved image to {image_file_path}")
        return True  # บันทึกสำเร็จ
    else:
        print(f"Image {image_file_path} already exists, skipping.")
        return False  # ไม่บันทึกเพราะไฟล์มีอยู่แล้ว

# ฟังก์ชันแสดงรูปภาพด้วย Matplotlib
def display_image(img_rgb):
    plt.imshow(img_rgb)
    plt.axis('off')  # ปิดแสดงแกน
    plt.show()

# ฟังก์ชันตรวจจับใบหน้าและวาดกรอบ
def detect_and_display(frame, frame_count, member_idx):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # วาดกรอบสี่เหลี่ยมรอบใบหน้าที่ตรวจพบ
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ข้อความที่จะแสดงในเฟรม
        member_name, folder_name = members[member_idx]
        text = f"{folder_name}_{member_name}_Frame:{frame_count}"

        # วางข้อความไว้ด้านบนของกรอบใบหน้า
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# เริ่มต้นสร้างโฟลเดอร์
create_folders()

# เปิดกล้องวิดีโอ
cap = cv2.VideoCapture(0)

frame_count = 1  # นับจำนวนเฟรม เริ่มจาก 1
current_member_idx = 0  # ตำแหน่งของสมาชิกที่จะบันทึกภาพ

while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # ตรวจจับใบหน้าและวาดกรอบพร้อมตำแหน่ง ชื่อ และเฟรมที่
    frame_with_faces = detect_and_display(frame, frame_count, current_member_idx)

    # แสดงเฟรม
    cv2.imshow('Video Stream', frame_with_faces)

    # รอการกดปุ่ม
    key = cv2.waitKey(1) & 0xFF

    # กด 'f' เพื่อบันทึกรูปภาพในเฟรมปัจจุบัน
    if key == ord('f'):
        saved = save_image_to_folders(frame_with_faces, frame_count, current_member_idx)
        if saved:  # ถ้าบันทึกสำเร็จ ให้เพิ่ม frame_count
            frame_count += 1

    # กด 'n' เพื่อไปยังสมาชิกถัดไป
    if key == ord('n'):
        current_member_idx = (current_member_idx + 1) % len(members)

    # กด 'q' เพื่อออกจากโปรแกรม
    if key == ord('q'):
        break

# ปิดวิดีโอและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
