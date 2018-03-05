import csv

big_m = ['1','3','5','7','8','10','12']
small_m = ['4','6','9','11']

leap_y = ['12','16']

def diff_day(date,pre_date):
    m, d, y = date.split('/')
    m_pre, d_pre, y_pre = pre_date.split('/')

    if m == m_pre:
        return int(d) - int(d_pre)
    else:
        if m_pre in big_m:
            return 31 - int(d_pre) + int(d)
        elif m_pre in small_m:
            return 30 - int(d_pre) + int(d)
        elif m_pre == '2':
            if y in leap_y:
                return 29 - int(d_pre) + int(d)
            else:
                return 28 - int(d_pre) + int(d)


pre_date = '1/1/10'
with open("prices.csv", "r") as f:
    reader = csv.reader(f)
    head_strings = next(reader)
    for i, line in enumerate(reader):
        date, symbol, open, close, high, low, volumn = tuple(line)
        if date != pre_date:
            diff = diff_day(date,pre_date)
            if not (diff == 3 or diff == 1):
                print(str(diff) + ' days intervals  pre: ' + pre_date + ' new: ' + date)
            pre_date = date
