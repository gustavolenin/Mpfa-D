import os

import pandas as pd
import numpy as np


def test_case(startswith, num):
    dfs = []
    files = [
        filename for filename in os.listdir() if
        filename.startswith(startswith) and
        'vtk' not in filename and
        f"test_case_{num}" in filename
    ]
    for file in files:
        with open(file, 'r') as res:
            data = res.read()
            key_value_pairs = [[pair.strip() for pair in line.split(':')] for line in data.split('\n') if line.strip() and ':' in line]
            d = {k.replace(' ', '_').replace('-', '_'): [float(v), ] for k, v in key_value_pairs}
        dfs.append(pd.DataFrame(d))
    df = pd.concat(dfs)
    df = df.sort_values("Unknowns")
    return df


def calculate_rates(df):
    df['q_l2_error'] = df['Relative_L2_norm'].apply(np.log).diff().apply(np.negative) \
                       / df['Unknowns'].apply(np.log).diff()
    df['q_grad_error'] = df['gradient_norm'].apply(np.log).diff().apply(np.negative) \
                      / df['Unknowns'].apply(np.log).diff()
    return df


if __name__ == "__main__":
    df1 = test_case('mge_', 1)
    df2 = test_case('mge_', 2)
    err1 = calculate_rates(df1)
    err2 = calculate_rates(df2)
    err1.to_csv('results1.csv')
    err2.to_csv('results2.csv')

