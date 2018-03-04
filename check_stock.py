import csv

stock_list = []
pre_date = ''
check_cnt = 0
with open("prices.csv", "r") as f:
    reader = csv.reader(f)
    head_strings = next(reader)
    for i, line in enumerate(reader):
        date, symbol, open, close, high, low, volumn = tuple(line)
        m,d,y = date.split('/')
        if date != pre_date:
            if check_cnt != 0:
                print('missing %d stock(s) in %s' % (check_cnt,pre_date))
            check_cnt = len(stock_list)
            pre_date = date
        if symbol not in stock_list:
            if date != '1/4/10':
                print(date + ' new stock: ' + symbol)
                stock_list.append(symbol)
        else:
            check_cnt -= 1





print(stock_list)
print(len(stock_list))



