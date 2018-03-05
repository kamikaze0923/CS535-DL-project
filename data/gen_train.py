import csv


pre_date = ''
full_line = []
newfile = open('data.csv','w')
with open('cropped_prices.csv', 'r') as f:
    reader = csv.reader(f)
    head_strings = next(reader)
    for i, line in enumerate(reader):
        date, symbol, open, close, high, low, volumn = tuple(line)
        if date != pre_date:
            if pre_date != '':
                strr = ','.join(full_line) + '\n'
                newfile.write(strr)
                full_line = []
            pre_date = date
        full_line += [open, close, high, low, volumn]

    strr = ','.join(full_line) + '\n'
    newfile.write(strr)