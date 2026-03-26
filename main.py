"""Entry point for Face Recognition Security System."""

import argparse
import sys
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Face Recognition Security System")
    parser.add_argument("--mode", choices=["api", "webcam", "register"], default="api")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--faces-dir", default="known_faces")
    parser.add_argument("--db", default="faces.db")
    parser.add_argument("--threshold", type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "api":
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=args.port, reload=False)

    elif args.mode == "webcam":
        from src.recognizer import FaceRecognizer
        from src.logger import AccessLogger

        recognizer = FaceRecognizer(db_path=args.db, threshold=args.threshold)
        logger = AccessLogger()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            sys.exit(1)

        print("Live recognition started. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = recognizer.recognize_frame(frame)
            labels = []
            for r in results:
                name = r["name"]
                conf = r["confidence"]
                labels.append(f"{name} ({conf:.2f})")
                logger.log(name, conf)

            faces = [r["face"] for r in results]
            annotated = recognizer.detector.draw_faces(frame, faces, labels)
            cv2.imshow("Face Recognition", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif args.mode == "register":
        from src.recognizer import FaceRecognizer

        recognizer = FaceRecognizer(db_path=args.db)
        faces_dir = Path(args.faces_dir)

        if not faces_dir.exists():
            print(f"Directory not found: {faces_dir}")
            sys.exit(1)

        for img_path in faces_dir.glob("*.[jp][pn][g]"):
            name = img_path.stem.replace("_", " ").title()
            image = cv2.imread(str(img_path))
            pid = recognizer.register_face(image, name)
            status = f"registered (ID: {pid})" if pid else "no face detected"
            print(f"{name}: {status}")


if __name__ == "__main__":
    main()
