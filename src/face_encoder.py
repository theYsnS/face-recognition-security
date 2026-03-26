"""Face encoding and comparison module."""

import numpy as np
import face_recognition


class FaceEncoder:
    """Generate and compare face embeddings."""

    def __init__(self, num_jitters: int = 1, model: str = "large"):
        self.num_jitters = num_jitters
        self.model = model  # "small" (5-point) or "large" (68-point)

    def encode(self, image: np.ndarray, known_locations: list | None = None) -> list[np.ndarray]:
        """Generate 128-d face encodings from image."""
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=known_locations,
            num_jitters=self.num_jitters,
            model=self.model,
        )
        return encodings

    def encode_single(self, image: np.ndarray) -> np.ndarray | None:
        """Encode a single face from image. Returns None if no face found."""
        encodings = self.encode(image)
        return encodings[0] if encodings else None

    def compare(
        self,
        known_encoding: np.ndarray,
        unknown_encoding: np.ndarray,
        threshold: float = 0.6,
    ) -> tuple[bool, float]:
        """Compare two face encodings. Returns (match, distance)."""
        distance = float(np.linalg.norm(known_encoding - unknown_encoding))
        return distance <= threshold, distance

    def batch_compare(
        self,
        known_encodings: list[np.ndarray],
        unknown_encoding: np.ndarray,
        threshold: float = 0.6,
    ) -> list[tuple[bool, float]]:
        """Compare unknown encoding against multiple known encodings."""
        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        return [(d <= threshold, float(d)) for d in distances]

    def find_best_match(
        self,
        known_encodings: list[np.ndarray],
        known_names: list[str],
        unknown_encoding: np.ndarray,
        threshold: float = 0.6,
    ) -> tuple[str | None, float]:
        """Find the closest matching face. Returns (name, distance) or (None, distance)."""
        if not known_encodings:
            return None, float("inf")

        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= threshold:
            return known_names[best_idx], best_distance
        return None, best_distance
