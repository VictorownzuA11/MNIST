from functools import reduce

def toHex(chunk):
    lst = []
    for ch in chunk:
        hv = hex(ch).replace('0x', '')
        if len(hv) == 1:
            hv = '0'+hv
        lst.append(hv)
    return reduce(lambda x,y:x+y, lst)

def printNum(chunk): 
    for i in range(0,28):
        print(toHex(chunk[28*i:28*(i+1)]))
    print()