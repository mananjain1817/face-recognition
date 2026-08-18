# Face Recognition System

A Python-based face recognition system that uses **DeepFace** and the **FaceNet512** model to identify a person in a test image by comparing their facial features with a dataset of known faces.

## Features

* Face detection using OpenCV
* Face recognition using DeepFace
* Face embeddings generated using FaceNet512
* Cosine distance for comparing facial embeddings
* Organized dataset with separate folders for each person
* Displays the best matching person and similarity information
* Uses a Python virtual environment for dependency management

## Technologies Used

* **Python 3.11**
* **OpenCV**
* **DeepFace**
* **TensorFlow**
* **FaceNet512**
* **SciPy**
* **NumPy**
* **Pandas**

## Project Structure

```text
Facerecog/
│
├── dataset/
│   ├── Angelina Jolie/
│   ├── Brad Pitt/
│   ├── Denzel Washington/
│   ├── Hugh Jackman/
│   ├── Jennifer Lawrence/
│   ├── Johnny Depp/
│   ├── Kate Winslet/
│   ├── Leonardo DiCaprio/
│   ├── Megan Fox/
│   ├── Natalie Portman/
│   ├── Nicole Kidman/
│   ├── Robert Downey Jr/
│   ├── Sandra Bullock/
│   ├── Scarlett Johansson/
│   ├── Tom Cruise/
│   ├── Tom Hanks/
│   └── Will Smith/
│
├── test/
│   └── test.jpg
│
├── main.py
│
└── venv/
```

## How It Works

The system follows these steps:

```text
Input Test Image
       ↓
Face Detection
       ↓
FaceNet512
       ↓
512-Dimensional Face Embedding
       ↓
Compare with Dataset Embeddings
       ↓
Calculate Cosine Distance
       ↓
Find Closest Match
       ↓
Display Recognized Person
```

### 1. Input Image

The image to be recognized is placed inside:

```text
test/test.jpg
```

### 2. Face Detection

OpenCV is used as the face detector through DeepFace.

```python
detector_backend="opencv"
```

### 3. Face Embedding

DeepFace uses the **FaceNet512** model to convert a detected face into a 512-dimensional numerical representation.

```python
model_name="Facenet512"
```

### 4. Face Comparison

The test embedding is compared with embeddings from the dataset using **cosine distance**.

A smaller cosine distance indicates that two facial embeddings are more similar.

For example, during testing:

```text
Test embedding: 512
Dataset embedding: 512
Cosine distance: 0.2277
```

This showed that the test image had a close match with the corresponding dataset image.

## Installation

Create and activate a virtual environment:

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\activate
```

Install the required packages:

```powershell
pip install deepface opencv-python scipy
```

## Running the Project

Make sure the virtual environment is activated:

```powershell
.\venv\Scripts\activate
```

Then run:

```powershell
python main.py
```

The program will process the test image and compare it with the faces stored in the dataset.

## Example Output

```text
Test image: C:\Project\Facerecog\test\test.jpg
Dataset: C:\Project\Facerecog\dataset
Image loaded successfully.
Test embedding generated: 512

===== FACE RECOGNITION RESULT =====

Recognized Person : Angelina Jolie
Distance          : 0.2277
Matched Image     : C:\Project\Facerecog\dataset\Angelina Jolie\001_fe3347c0.jpg
```



## Important Notes

* The first execution can take considerable time because the facial embeddings may need to be generated for the dataset.
* The project currently runs primarily on the CPU.
* Recognition accuracy depends on image quality, lighting, face angle, and the quality of the dataset.
* Cosine distance is used as a similarity measure; it should not automatically be interpreted as a probability or guaranteed confidence percentage.
* The `venv` folder should generally **not be uploaded to GitHub**.

## Future Improvements

* Create a persistent embedding cache to avoid recalculating dataset embeddings.
* Add real-time face recognition using a webcam.
* Display the recognized person's name directly on the video feed.
* Add a graphical user interface.
* Improve handling of multiple faces in a single image.
* Add an "Unknown Person" threshold.
* Improve performance using optimized embedding storage and retrieval.

 Dataset used: https://www.kaggle.com/datasets/bhaveshmittal/celebrity-face-recognition-dataset


