import os
import numpy as np
import pandas as pd


curr_folder = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    a = np.random.rand(1000, 10)
    df = pd.DataFrame(a).assign(label=list(range(5)) * 200)
    df.to_csv('test_full_digits.csv')
    df.astype('float16').to_csv('test_half_digits.csv')
    print(os.path.getsize('test_full_digits.csv'))
    print(os.path.getsize('test_half_digits.csv'))
