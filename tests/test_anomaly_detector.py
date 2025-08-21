import os
import unittest
import pandas as pd
import numpy as np
from src.anomaly_detector import AnomalyDetector

def create_dummy_data(train_file, test_file):
    """Creates dummy CSV files for testing."""
    train_data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='min')),
        'value': np.random.rand(100) * 100
    }
    pd.DataFrame(train_data).to_csv(train_file, index=False)

    test_data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-02', periods=50, freq='min')),
        'value': np.random.rand(50) * 100
    }
    pd.DataFrame(test_data).to_csv(test_file, index=False)

    # Introduce an anomaly for a more realistic test
    test_data['value'][25] = 1000  # A clear outlier
    pd.DataFrame(test_data).to_csv(test_file, index=False)

class TestAnomalyDetector(unittest.TestCase):
    """Test suite for the AnomalyDetector class."""

    @classmethod
    def setUpClass(cls):
        """Create dummy data files for all tests in this class."""
        cls.train_file = "test_train.csv"
        cls.test_file = "test_test.csv"
        cls.pred_file = "test_pred.csv"
        create_dummy_data(cls.train_file, cls.test_file)

    @classmethod
    def tearDownClass(cls):
        """Clean up dummy files after all tests are finished."""
        os.remove(cls.train_file)
        os.remove(cls.test_file)
        if os.path.exists(cls.pred_file):
            os.remove(cls.pred_file)

    def test_initialization(self):
        """Test that the class initializes correctly with valid files."""
        detector = AnomalyDetector(self.train_file, self.test_file, self.pred_file)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.train_file, self.train_file)
        self.assertEqual(detector.test_file, self.test_file)

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            AnomalyDetector("non_existent_train.csv", "non_existent_test.csv", "pred.csv")

    def test_full_pipeline(self):
        """Test the train, predict, and save methods work in sequence."""
        detector = AnomalyDetector(self.train_file, self.test_file, self.pred_file)
        detector.train()
        self.assertIsNotNone(detector.model)

        # Check that a prediction report is generated
        detector.predict(alpha=0.05)
        self.assertTrue(os.path.exists(self.pred_file))

        # Load the output file and check for the 'anomaly' column
        df_pred = pd.read_csv(self.pred_file)
        self.assertIn('anomaly', df_pred.columns)
        self.assertTrue(df_pred['anomaly'].any(), "No anomalies detected.")

    def test_load_and_save_model(self):
        """Test that a trained model can be saved and loaded."""
        model_path = "test_model.pkl"
        detector1 = AnomalyDetector(self.train_file, self.test_file, self.pred_file)
        detector1.train()
        detector1.save(model_path)
        self.assertTrue(os.path.exists(model_path))

        detector2 = AnomalyDetector(self.train_file, self.test_file, self.pred_file)
        detector2.load(model_path)
        self.assertIsNotNone(detector2.model)

        # Clean up the saved model
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()

# python -m unittest tests/test_anomaly_detector.py
