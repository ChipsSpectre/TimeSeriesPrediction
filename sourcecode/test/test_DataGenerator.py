import unittest

from src.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    def test_sine(self):
        default_ts = DataGenerator.generate_sine()

        expected_len = 10000
        self.assertEquals(default_ts.shape[0], expected_len)
        self.assertAlmostEqual(default_ts[0], 0.0)
        self.assertAlmostEqual(default_ts[-1], 0.0)

        middle = expected_len // 2
        self.assertAlmostEqual(default_ts[middle], 1.0)


if __name__ == '__main__':
    unittest.main()
