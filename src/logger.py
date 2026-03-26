"""Access logging for recognition events."""

import csv
from datetime import datetime
from pathlib import Path


class AccessLogger:
    """Log face recognition events."""

    def __init__(self, log_file: str = "logs/access_log.csv"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "person", "confidence", "camera_id"])

    def log(self, person: str, confidence: float, camera_id: str = "default") -> None:
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), person, f"{confidence:.3f}", camera_id])

    def get_recent(self, limit: int = 100) -> list[dict]:
        if not self.log_file.exists():
            return []
        entries = []
        with open(self.log_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
        return entries[-limit:]
