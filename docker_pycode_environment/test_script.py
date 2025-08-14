import pandas as pd
import numpy as np

def main():
    print("Executing the sandboxed Python script...")
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    print("Pandas DataFrame created:")
    print(df)
    print("Numpy array created:")
    print(np.array([5, 6, 7]))

if __name__ == "__main__":
    main()