import streamlit as st
import os
import json
from datetime import datetime
from PIL import Image

# ---------------------------
# FIX: PyTorch YOLOv8 Safe Loading (WICHTIG)
# ---------------------------
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

# ---------------------------
# FIX: YOLO CONFIG DIR (Streamlit Cloud)
# ---------------------------
os.environ["YOLO_CONFIG_DIR"] = "/tmp"

# ---------------------------
# YOLOv8 MODEL
# ---------------------------
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # lädt automatisch oder cached
    return model

model = load_model()

# ---------------------------
# CONFIG
# ---------------------------
UPLOAD_FOLDER = "uploads"
DB_FILE = "fundbuero.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# DATABASE
# ---------------------------
def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return []

def save_database(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ---------------------------
# UI
# ---------------------------
st.title("🏫 Digitales Fundbüro (YOLOv8)")

menu = st.sidebar.selectbox(
    "Menü",
    ["Gegenstand hochladen", "Durchsuchen"]
)

# ===========================
# UPLOAD
# ===========================
if menu == "Gegenstand hochladen":
    st.header("📸 Neuen Gegenstand melden")

    finder_name = st.text_input("Dein Name")
    location = st.text_input("Fundort")
    description = st.text_area("Beschreibung (optional)")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        class_name = "Unbekannt"
        confidence_score = 0.0

        if st.button("🔍 Kategorie erkennen"):
            results = model(image)

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence_score = float(box.conf[0])

                st.success(f"Erkannt: {class_name}")
                st.write(f"Sicherheit: {round(confidence_score * 100, 2)} %")

                st.image(results[0].plot(), caption="Erkennung")

            else:
                st.warning("Kein Objekt erkannt.")

        if st.button("💾 Speichern"):
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image.save(filepath)

            db = load_database()
            db.append({
                "id": len(db) + 1,
                "category": class_name,
                "confidence": confidence_score,
                "image_path": filepath,
                "finder": finder_name,
                "location": location,
                "description": description,
                "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "status": "Offen",
                "messages": []
            })

            save_database(db)
            st.success("Gespeichert!")

# ===========================
# SUCHEN
# ===========================
elif menu == "Durchsuchen":
    st.header("🔎 Fundstücke")

    db = load_database()

    if not db:
        st.info("Keine Einträge vorhanden.")
    else:
        categories = list(set(item["category"] for item in db))
        selected_category = st.selectbox("Kategorie", ["Alle"] + categories)

        status_filter = st.selectbox("Status", ["Alle", "Offen", "Abgeholt"])

        for item in db:
            if (selected_category == "Alle" or item["category"] == selected_category) and \
               (status_filter == "Alle" or item["status"] == status_filter):

                st.image(item["image_path"], width=250)
                st.write(f"📂 {item['category']}")
                st.write(f"📍 {item['location']}")
                st.write(f"📝 {item['description']}")
                st.write(f"👤 {item['finder']}")
                st.write(f"📅 {item['date']}")
                st.write(f"📌 {item['status']}")

                with st.expander("📬 Nachricht"):
                    msg = st.text_area("Nachricht", key=f"msg_{item['id']}")
                    if st.button("Senden", key=f"send_{item['id']}"):
                        item["messages"].append({
                            "text": msg,
                            "date": datetime.now().strftime("%d.%m.%Y %H:%M")
                        })
                        save_database(db)
                        st.success("Gesendet!")

                if item["status"] == "Offen":
                    if st.button("✅ Abgeholt", key=f"done_{item['id']}"):
                        item["status"] = "Abgeholt"
                        save_database(db)
                        st.success("Status aktualisiert!")

                st.write("---")
