"""Face detection module using face_recognition library."""

from typing import Optional

import cv2
import face_recognition
import numpy as np


class FaceDetector:
    """Detect and locate faces in images/frames."""

    def __init__(self, model: str = "hog", upsample: int = 1):
        self.model = model  # "hog" (CPU) or "cnn" (GPU)
        self.upsample = upsample

    def detect_faces(self, frame: np.ndarray) -> list[dict]:
        """Detect faces and return locations with landmarks."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(
            rgb, model=self.model, number_of_times_to_upsample=self.upsample
        )
        landmarks = face_recognition.face_landmarks(rgb, locations)

        faces = []
        for i, (top, right, bottom, left) in enumerate(locations):
            face_img = rgb[top:bottom, left:right]
            faces.append({
                "location": (top, right, bottom, left),
                "bbox": (left, top, right, bottom),
                "face_image": face_img,
                "landmarks": landmarks[i] if i < len(landmarks) else None,
            })
        return faces

    def align_face(
        self, frame: np.ndarray, face: dict, target_size: tuple[int, int] = (160, 160)
    ) -> np.ndarray:
        """Crop and align face for encoding."""
        top, right, bottom, left = face["location"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        face_img = rgb[top:bottom, left:right]
        return cv2.resize(face_img, target_size)

    def draw_faces(
        self, frame: np.ndarray, faces: list[dict], labels: Optional[list[str]] = None
    ) -> np.ndarray:
        """Draw bounding boxes around detected faces."""
        annotated = frame.copy()
        for i, face in enumerate(faces):
            left, top, right, bottom = face["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            if labels and i < len(labels):
                cv2.putText(
                    annotated, labels[i], (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
        return annotated
