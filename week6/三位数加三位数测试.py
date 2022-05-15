import random
import time

def text():
    a = random.randint(100,999)
    b = random.randint(100,999)
    c = a + b
    c_try = eval(input(str(a) + '+' + str(b) + '='))
    if c == c_try:
        print('正确，您太强了')
        return True
    else:
        print('错误，您是菜鸡')
        return False

input('回车键开始')
score = 0
time_start = time.time()
for i in range(15):
    print('第'+str(i+1)+'题')
    tf = text()
    if tf == True:
        score += 1
    else:
        score += 0
time_end = time.time()
print('您答对了'+str(score)+'题')
print('耗时'+str(time_end-time_start)+'s')
input('回车退出QAQ')
