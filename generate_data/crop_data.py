import csv

stock_list = []
ori_list = []
pre_date = ''
check_cnt = 0
record_flag = False

newfile = open('cropped_prices.csv','w')
with open('prices.csv', 'r') as f:
    reader = csv.reader(f)
    head_strings = next(reader)
    newfile.write('%s,%s,%s,%s,%s,%s,%s\n' % tuple([i for i in head_strings]))
    for i, line in enumerate(reader):
        date, symbol, open, close, high, low, volumn = tuple(line)
        if record_flag == False:
            if date != '1/2/13':
                continue
            else:
                record_flag = True
        if date == '1/2/13':
            ori_list.append(symbol)
        if symbol not in stock_list:
            if date != '1/2/13':
                print(date + ' new stock: ' + symbol)
            stock_list.append(symbol)
        if symbol in ori_list:
            newfile.write('%s,%s,%s,%s,%s,%s,%s\n' % (date, symbol, open, close, high, low, volumn))

newfile.close()


# print(stock_list)
# print(ori_list)
# print(len(stock_list))
# print(len(ori_list))




