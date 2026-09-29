import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import predict


class TicketClassifierRegressionTests(unittest.TestCase):
    def test_predict_ticket_trains_when_model_is_missing(self):
        model_path = ROOT / "model" / "ticket_classifier.pkl"
        backup_path = ROOT / "model" / "ticket_classifier.pkl.bak"

        if backup_path.exists():
            backup_path.unlink()
        if model_path.exists():
            model_path.rename(backup_path)

        try:
            category, confidence = predict.predict_ticket("I forgot my password!!!")
            self.assertEqual(category, "Login Issue")
            self.assertGreaterEqual(confidence, 0)
            self.assertLess(confidence, 100)
        finally:
            if model_path.exists():
                model_path.unlink()
            if backup_path.exists():
                backup_path.rename(model_path)


if __name__ == "__main__":
    unittest.main()
