"""Tests for face recognition system."""

import os
import unittest
from src.database import FaceDatabase
import numpy as np


class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        self.db = FaceDatabase("test_faces.db")

    def tearDown(self):
        if os.path.exists("test_faces.db"):
            os.remove("test_faces.db")

    def test_add_and_get_person(self):
        encoding = np.random.rand(128)
        pid = self.db.add_person("Test User", encoding, "admin")
        person = self.db.get_person(pid)
        self.assertEqual(person["name"], "Test User")
        self.assertEqual(person["role"], "admin")

    def test_get_all_persons(self):
        self.db.add_person("User 1", np.random.rand(128))
        self.db.add_person("User 2", np.random.rand(128))
        persons = self.db.get_all_persons()
        self.assertEqual(len(persons), 2)

    def test_delete_person(self):
        pid = self.db.add_person("Delete Me", np.random.rand(128))
        self.assertTrue(self.db.delete_person(pid))
        self.assertIsNone(self.db.get_person(pid))

    def test_get_all_encodings(self):
        self.db.add_person("User", np.random.rand(128))
        encodings, names, ids = self.db.get_all_encodings()
        self.assertEqual(len(encodings), 1)
        self.assertEqual(names[0], "User")

    def test_count(self):
        self.db.add_person("A", np.random.rand(128))
        self.db.add_person("B", np.random.rand(128))
        self.assertEqual(self.db.count(), 2)


if __name__ == "__main__":
    unittest.main()
