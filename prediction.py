import csv
import numpy


with open('cropped_prices.csv', 'r') as f:
    reader = csv.reader(f)
    head_strings = next(reader)
    for i, line in enumerate(reader):
        date, symbol, open, close, high, low, volumn = tuple(line)
