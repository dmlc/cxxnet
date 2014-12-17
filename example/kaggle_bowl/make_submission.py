import csv
import sys

if len(sys.argv) < 4:
    print "Usage: python make_submission.py sample_submission.csv test.lst text.txt out.csv"
    exit(1)

fc = csv.reader(file(sys.argv[1]))
fl = csv.reader(file(sys.argv[2]), delimiter='\t', lineterminator='\n')
fi = csv.reader(file(sys.argv[3]), delimiter=' ', lineterminator='\n')
fo = csv.writer(open(sys.argv[4], "w"), lineterminator='\n')

head = fc.next()
fo.writerow(head)

head = head[1:]

img_lst = []
for line in fl:
    path = line[-1]
    path = path.split('/')
    path = path[-1]
    img_lst.append(path)

idx = 0
for line in fi:
    row = [img_lst[idx]]
    idx += 1
    line = line[:-1]
    row.extend(line)
    fo.writerow(row)

    

