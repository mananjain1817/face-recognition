import os
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine

# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(BASE_DIR, "test", "test.jpg")
db_path = os.path.join(BASE_DIR, "dataset")

print("Test image:", img_path)
print("Dataset:", db_path)

# ==============================
# CHECK FILES
# ==============================

if not os.path.isfile(img_path):
    print("ERROR: test.jpg not found")
    exit()

if not os.path.isdir(db_path):
    print("ERROR: dataset folder not found")
    exit()

# ==============================
# LOAD TEST IMAGE
# ==============================

img = cv2.imread(img_path)

if img is None:
    print("ERROR: Could not load test image")
    exit()

print("Image loaded successfully.")

# ==============================
# GENERATE TEST EMBEDDING
# ==============================

test_embedding = DeepFace.represent(
    img_path=img_path,
    model_name="Facenet512",
    detector_backend="opencv",
    enforce_detection=True
)[0]["embedding"]

print("Test embedding generated:", len(test_embedding))

# ==============================
# SEARCH DATASET
# ==============================

best_match = None
best_distance = float("inf")

valid_extensions = (".jpg", ".jpeg", ".png")

for person in os.listdir(db_path):

    person_path = os.path.join(db_path, person)

    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):

        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(person_path, filename)

        try:

            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                detector_backend="opencv",
                enforce_detection=True
            )[0]["embedding"]

            distance = cosine(test_embedding, embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = (person, image_path)

        except Exception:
            continue

# ==============================
# RESULT
# ==============================

print("\n===== FACE RECOGNITION RESULT =====")

if best_match is None:

    print("No valid face matches found.")

else:

    person, image_path = best_match

    print("Recognized Person :", person)
    print("Distance          :", round(best_distance, 4))
    print("Matched Image     :", image_path)

    # Simple similarity score
    similarity = max(0, (1 - best_distance) * 100)

    print("Similarity        :", round(similarity, 2), "%")