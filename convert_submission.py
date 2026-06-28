import os

import pandas as pd


def main():
    csv_path = "submission.csv"
    xlsx_path = "submission.xlsx"

    if os.path.exists(csv_path):
        # Read the ranked candidate output CSV
        df = pd.read_csv(csv_path)
        # Convert and save as Excel (XLSX) format
        df.to_excel(xlsx_path, index=False)
        print(f"Successfully converted {csv_path} to {xlsx_path}")
    else:
        print(f"Error: {csv_path} does not exist.")


if __name__ == "__main__":
    main()
