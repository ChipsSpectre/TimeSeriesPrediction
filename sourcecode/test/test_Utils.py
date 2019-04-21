import numpy as np
import unittest

from src.utils import train_val_split

class TestUtils(unittest.TestCase):

    def test_train_val_split(self):
        N = 100
        data = np.zeros(N)

        with self.assertRaises(ValueError):
            train_val_split(data, -1)        
        with self.assertRaises(ValueError):
            train_val_split(data, 2)        
        with self.assertRaises(ValueError):
            train_val_split(data, 0.5, axis=1)        

        train, val = train_val_split(data, 0)
        self.assertEquals(train.shape[0], N)
        self.assertEquals(val.shape[0], 0)   

        train, val = train_val_split(data, 1)
        self.assertEquals(train.shape[0], 0)
        self.assertEquals(val.shape[0], N)   

        train, val = train_val_split(data, 0.5)
        self.assertEquals(train.shape[0], N//2)
        self.assertEquals(val.shape[0], N//2)

        fraction = 0.50001 # slightly more than the half for validation
        train, val = train_val_split(data, fraction)
        self.assertEquals(train.shape[0], int((1-fraction) * N))
        self.assertEquals(val.shape[0], N - int((1-fraction) * N))

        

if __name__ == '__main__':
    unittest.main()