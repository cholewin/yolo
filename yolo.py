import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import json
from datetime import datetime

# ---------------------------
# CONFIG
# ---------------------------
UPLOAD_FOLDER = "uploads"
DB_FILE = "fundbuero.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# MODEL LADEN (YOLOv8)
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Versuch: Modell von Hugging Face laden
        model_path = hf_hub_download(
            repo_id="Ultralytics/YOLOv8",
            filename="yolov8n.pt"
        )
    except:
        # Fallback: direkt laden (falls HF nicht geht)
        model_path = "yolov8n.pt"

    model = YOLO(model_path)
    return model

model = load_model()

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
st.title("🏫 Digitales Fundbüro")

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
    location = st.text_input("Fundort (z.B. Aula, Sporthalle)")
    description = st.text_area("Beschreibung (optional)")

    uploaded_file = st.file_uploader("Foto hochladen", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Kategorie automatisch erkennen"):

            results = model(image)

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence_score = float(box.conf[0])

                st.success(f"Erkannte Kategorie: {class_name}")
                st.write(f"Sicherheit: {round(confidence_score*100, 2)} %")

                # Bild mit Bounding Box anzeigen
                result_img = results[0].plot()
                st.image(result_img, caption="Erkennung", use_container_width=True)

            else:
                class_name = "Unbekannt"
                confidence_score = 0.0
                st.warning("Kein Objekt erkannt.")

            # Speichern Button
            if st.button("Im Fundbüro speichern"):
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
                st.success("Gegenstand wurde gespeichert!")

# ===========================
# SUCHEN
# ===========================
elif menu == "Durchsuchen":
    st.header("🔎 Gefundene Gegenstände")

    db = load_database()

    if len(db) == 0:
        st.info("Noch keine Gegenstände vorhanden.")
    else:
        categories = list(set(item["category"] for item in db))
        selected_category = st.selectbox("Kategorie", ["Alle"] + categories)

        status_filter = st.selectbox("Status", ["Alle", "Offen", "Abgeholt"])

        for item in db:
            if (selected_category == "Alle" or item["category"] == selected_category) and \
               (status_filter == "Alle" or item["status"] == status_filter):

                st.image(item["image_path"], width=250)
                st.write(f"📂 Kategorie: {item['category']}")
                st.write(f"📍 Fundort: {item['location']}")
                st.write(f"📝 Beschreibung: {item['description']}")
                st.write(f"👤 Finder: {item['finder']}")
                st.write(f"📅 Datum: {item['date']}")
                st.write(f"📌 Status: {item['status']}")

                # Nachrichten
                with st.expander("📬 Nachricht hinterlassen"):
                    message = st.text_area(f"Nachricht für {item['finder']}", key=f"msg_{item['id']}")
                    if st.button("Senden", key=f"send_{item['id']}"):
                        item["messages"].append({
                            "text": message,
                            "date": datetime.now().strftime("%d.%m.%Y %H:%M")
                        })
                        save_database(db)
                        st.success("Nachricht gespeichert!")

                if item["messages"]:
                    with st.expander("📨 Nachrichten anzeigen"):
                        for msg in item["messages"]:
                            st.write(f"{msg['date']} - {msg['text']}")

                if item["status"] == "Offen":
                    if st.button("✅ Als abgeholt markieren", key=f"done_{item['id']}"):
                        item["status"] = "Abgeholt"
                        save_database(db)
                        st.success("Status geändert!")

                st.write("---")
