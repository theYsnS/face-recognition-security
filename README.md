# Face Recognition Security System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered face recognition system for security and access control. Features real-time face detection, recognition with persistent identity database, REST API for integration, and comprehensive access logging.

## Features

- **Real-time Detection**: MTCNN/HOG-based face detection from camera feeds
- **Face Recognition**: 128-dimensional face encoding with configurable matching threshold
- **Identity Database**: SQLite-backed face database with CRUD operations
- **REST API**: FastAPI endpoints for registration, recognition, and log access
- **Access Logging**: Timestamped logs of all recognition events with CSV export
- **Multi-mode**: Webcam live mode, API server mode, batch registration mode

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐
│  Camera   │────▶│ FaceDetector │────▶│ FaceEncoder  │
│  / Image  │     │ (MTCNN/HOG)  │     │ (128-d)      │
└──────────┘     └──────────────┘     └──────┬───────┘
                                              │
┌──────────┐     ┌──────────────┐     ┌───────▼──────┐
│  Logger   │◀───│  Recognizer  │◀───▶│   Database   │
│  (CSV)    │     │  (Matching)  │     │  (SQLite)    │
└──────────┘     └──────────────┘     └──────────────┘
                        │
                  ┌─────▼─────┐
                  │  FastAPI   │
                  │  REST API  │
                  └───────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Register a new face with name and image |
| POST | `/recognize` | Recognize faces in uploaded image |
| GET | `/persons` | List all registered persons |
| DELETE | `/persons/{id}` | Remove a registered person |
| GET | `/logs` | Get access/recognition logs |

## Installation

```bash
git clone https://github.com/theYsnS/face-recognition-security.git
cd face-recognition-security
pip install -r requirements.txt
```

## Usage

```bash
# Start API server
python main.py --mode api --port 8000

# Live webcam recognition
python main.py --mode webcam

# Register faces from directory
python main.py --mode register --faces-dir ./known_faces/
```

## License

MIT License - see [LICENSE](LICENSE) for details.
