"""
Take dataset as parameter, display info for all numerical features.
Info to display: count, mean, std, min, 25%, 50%, 75%, Max.
Each feature gets a column, each info gets a row.
Cannot use function that calculates info by itself.
"""
import sys
import csv

def main():
    try:
        assert len(sys.argv) == 2, "wrong argument count"
        assert sys.argv[1].endswith('.csv'), "wrong file type"
        with open(sys.argv[1]) as data:
            content = csv.reader(data, delimiter=' ', quotechar='|')
            for row in content:
                print(', '.join(row))
    except AssertionError as error:
        print("AssertionError:", error)

if __name__ == "__main__":
    main()