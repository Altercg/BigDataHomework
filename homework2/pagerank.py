# 迭代法
import numpy as np

d = 0.8
n = 875713          # 数目和编号不是对应的
out = {}            # 记录各个顶点有的出度边
out_num = {}        # 记录各个定点的出度数
r0 = {} # 前
r1 = {} # 后
epsilon = 0.00001

def cal():
    # 初始化
    global r0, r1, epsilon, out, out_num, n, d    
    e2 = 1
    while e2 > epsilon:
        for key in r0.keys():   # 初始化
            r1[key] = 0
        for node in out.keys():     # node页面
            for i in out[node]:     # node页面的出度页面i
                r1[i] += d*(1/out_num[node])*r0[node]    # d*M*r_t
        
        e2 = 0
        for k in r1.keys():
            r1[k] += (1-d)/n              # 还有b没有加上
            e2 = max(abs(r1[k] - r0[k]), e2)
        r0 = r1.copy()
        print(e2)

def loadData(filename):
    f = open(filename)
    for line in f.readlines():
        if line[0] == '#':  # 跳过数据集开头的说明部分
            continue
        line = line.strip('\n')
        linearr = line.strip().split('\t')
        # 处理out和out_num
        if linearr[0] not in out.keys():
            out[linearr[0]] = [linearr[1]]
            out_num[linearr[0]] = 1
        else:
            out[linearr[0]].append(linearr[1])
            out_num[linearr[0]] += 1
        # 处理r0
        for i in range(2):
            if linearr[i] not in r0.keys():
                r0[linearr[i]] = 1/n
    # m = pd.DataFrame(np.zeros([Nodes,Nodes]), index=index, colums=colums) # 存不下
    # print(m)

def output():
    print("output top 100 to file")
    f = open('reslut2.txt', 'w')
    r0_order = sorted(r0.items(), key=lambda x:x[1],reverse=True)
    k = 0
    for node in r0_order:     # node页面的出度页面i
        if k == 100:
            print("finish")
            break 
        f.write(node[0]+ ' ' + str(node[1]) + '\n')
        k += 1

if __name__ == '__main__':
    filename = 'web-Google.txt'
    loadData(filename)
    print("The data is loaded and the calculation begins")
    cal()
    output()