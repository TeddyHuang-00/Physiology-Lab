import time
import random
import os

def text(T):
    a = random.randint(1000000,9999999)
    print(a)
    time.sleep(T)
    os.system('cls')
    a_try = eval(input("请输入您刚才看到的数字:"))
    if a == a_try:
        print('正确，您太强了')
        return True
    else:
        print('错误，您和金鱼没有差别')
        return False


T = eval(input('请输入记忆时间T秒'))
# print(str(T)+"秒有这么长")
# time.sleep(T)
print('开始')
score = 0
for i in range(15):
    print("第"+str(i+1)+"题")
    tf = text(T)
    if tf == True:
        score += 1
    else:
        score += 0
print('您答对了'+str(score)+'题')
input("回车关掉我QAQ")