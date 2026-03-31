import csv
import sys

def count_above_tau(csv_file, tau):
    total_rows = 0
    above_rows = 0

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            p_z = float(row["p_z"])
            if p_z > tau:
                above_rows += 1

    percentage = (above_rows / total_rows * 100) if total_rows > 0 else 0.0

    print(f"Rows with p_z > {tau}: {above_rows}/{total_rows}")
    print(f"Percentage above tau: {percentage:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python count_pz.py <csv_file> <tau>")
        sys.exit(1)

    csv_file = sys.argv[1]
    tau = float(sys.argv[2])
    count_above_tau(csv_file, tau)