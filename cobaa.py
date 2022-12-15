import re
import random
import string

print("masukkan huruf : ")



def nama():
    # arr = []
    # sta = string.ascii_lowercase
    # for i in range(1, 10):
    #     arr.append("skill: kuli{}, -{:0.2f}%".format(random.choice(sta), random.random()))
    # print(arr)
    # print('\n')
    # print(sorted(arr, key=lambda s: float(re.search(r'(\d+)\.', s).groups()[0])))
    # print('\n')
    # print(sorted(arr, reverse=True))
    da = 120
    count = 0
    for i, count in enumerate(da):
        if count == 5:
            continue
        print(count)



nama()
