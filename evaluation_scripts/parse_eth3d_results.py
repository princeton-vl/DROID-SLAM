import re
import sys
import numpy as np

def extract_rmse_from_file(filepath):
    rmse_values = []
    # Regular expression to match 'rmse': <float>
    rmse_pattern = re.compile(r"'rmse':\s*([\d\.eE+-]+)")

    with open(filepath, 'r') as file:
        for line in file:
            match = rmse_pattern.search(line)
            if match:
                try:
                    rmse = float(match.group(1))
                    rmse_values.append(rmse)
                except ValueError:
                    print(f"Skipping invalid float: {match.group(1)}")

    return rmse_values


# Example usage:
filepath = sys.argv[1]
rmse_list = extract_rmse_from_file(filepath)

arr = 100 * np.array(rmse_list)
print("rmse auc 2cm", np.sum(np.clip(2.0 - arr, 0.0, None)))
print("rmse auc 8cm", np.sum(np.clip(8.0 - arr, 0.0, None)))
