import os
import random
lst = os.listdir("train")

index = 0
def writer(obj, name):
    w = open(name, "w")
    for item in obj:
        l = item.split(".")
        pid = l[1]
        tid = "0"
        if l[0] == "cat":
            tid = "0"
        else:
            tid = "1"
        global index
        index += 1
        line = '\t'.join([str(index), tid, item])
        w.write(line + '\n')
    w.close()


cats = []
dogs = []

for item in lst:
    if item[:3] == "cat":
        cats.append(item)
    else:
        dogs.append(item)

length = min(len(cats), len(dogs))
L = int(length * 0.7)

train = []
train.extend(cats[:L])
train.extend(dogs[:L])

test = []
test.extend(cats[L:])
test.extend(dogs[L:])

random.shuffle(train)
random.shuffle(test)

writer(train, "train.lst")
writer(test, "test.lst")
