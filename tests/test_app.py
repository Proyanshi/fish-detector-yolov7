import unittest
from src import app

class TestApp(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(app)

if __name__ == "__main__":
    unittest.main()
