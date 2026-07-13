import csv
from datetime import datetime
from collections import defaultdict
import sys

def calculate_rps(input_file):
    rps_counts = defaultdict(int)

    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader) 

        for row in reader:
            if not row:
                continue
            timestamp_str = row[0].strip()
            try:
                dt = datetime.fromisoformat(timestamp_str)
                second_key = int(dt.timestamp())
                rps_counts[second_key] += 1
            except ValueError as e:
                print(f"Analyzing Time stamp fail: '{timestamp_str}': {e}")
                continue

    if not rps_counts:
        print("No valid data")
        return

    avg_rps = sum(rps_counts.values()) / len(rps_counts)
    max_rps = max(rps_counts.values())
    min_rps = min(rps_counts.values())
    print(f"Average RPS: {avg_rps:.2f}")
    print(f"Max RPS: {max_rps}")
    print(f"Min RPS: {min_rps}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python rps_calculator.py <input.csv>")
        sys.exit(1)
    calculate_rps(sys.argv[1])

# Average RPS: 27.89
# Max RPS: 146
# Min RPS: 1