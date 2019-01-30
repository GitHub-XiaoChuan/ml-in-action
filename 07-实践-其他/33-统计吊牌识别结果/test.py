import xlrd
import glob

path = '/Users/xingoo/Documents/dataset/天狗-吊牌/images'

normal = 0
unnormal = 0

for i in range(1, 41):
    file_path = path + '/' + str(i) + '/' + str(i) + '.xls'
    print(file_path)

    sheet = xlrd.open_workbook(file_path).sheet_by_index(0)

    for i in range(1, sheet.nrows):
        if sheet.cell(i, 1).value == '正常':
            normal += 1
        else:
            unnormal += 1

print(normal)
print(unnormal)
