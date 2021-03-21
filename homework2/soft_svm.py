import numpy as np

n_train = 32561     # 训练集的个数
n_test = 16281          # 测试集的个数
x_dimension = 123   # 观察数据发现特征维度
precision = 0.01    # 梯度的精度

def loadDataSet(filename):
    global x_dimension, n_train
    dataset = np.zeros((n_train, x_dimension))
    labelset = np.zeros((n_train, 1))
    f = open(filename)
    count = 0
    for line in f.readlines():
        linearr = line.strip().split(' ')
        # 标签
        if linearr[0] == '-1':
            labelset[count] = -1
        else:
            labelset[count] = 1
        # 特征
        for i in range(1, len(linearr)):
            featurearr = linearr[i].strip().split(':')
            dataset[count][int(featurearr[0])-1] = int(featurearr[1])
        count += 1
    x0 = np.ones((n_train, 1))
    dataset = np.hstack([x0, dataset])# 添加b，方便计算
    # 划分训练集和测试集
    train_indices = np.random.choice(dataset.shape[0], round(dataset.shape[0]*0.8), replace=False)
    validate_indices = np.array(list(set(range(dataset.shape[0]))-set(train_indices)))
    # 最终的数据
    train_data = np.array(dataset[train_indices])     # 训练
    train_label = np.array(labelset[train_indices])
    validate_data = np.array(dataset[validate_indices])   # 测试
    validate_label = np.array(labelset[validate_indices])
    return train_data, train_label, validate_data, validate_label

def loadTestSet(filename):
    global x_dimension, n_test
    dataset = np.zeros((n_test, x_dimension))
    f = open(filename)
    count = 0
    for line in f.readlines():
        linearr = line.strip().split(' ')
        # 特征
        for i in range(0, len(linearr)):
            featurearr = linearr[i].strip().split(':')
            dataset[count][int(featurearr[0])-1] = int(featurearr[1])
        count += 1
    x0 = np.ones((n_test, 1))
    dataset = np.hstack([x0, dataset])# 添加b，方便计算
    return dataset

class SoftMarginSVM(object):
    def __init__(self, train_data, train_label, validate_data, validate_label, alpha=0.00001, C=700.0):
        self.train_data = train_data
        self.train_label = train_label
        self.validate_data = validate_data
        self.validate_label = validate_label
        self.w = np.zeros(x_dimension+1)
        self.C = C                      # 惩罚参数      
        self.alpha = alpha              # 步长

    # def learning_rate(self, t, t0=100):     # t0 = 100 ,alpha=0.01
    #        return self.alpha/(t+t0)

    # 随机梯度下降
    def fit(self, n_iters=10):
        for cur_iter in range(n_iters):
            # i = np.random.choice(len(X))    # 选取某一个样本
            # 计算梯度
            for i in range(self.train_data.shape[0]):
                # 判断
                if self.train_label[i]*(self.train_data[i].dot(self.w)) >= 1:
                    diff = 0 
                else :
                    diff = -self.train_label[i]*self.train_data[i]    # 各维度一起计算
                # 更新
                diff = self.w + (self.C * diff)
                #self.w = self.w - self.learning_rate(cur_iter*self.train_data.shape[0]+i) * diff
                self.w = self.w - self.alpha*diff   # alpha = 0.00001
            # 检验w的正确率
            correct_num = 0
            for i in range(len(self.validate_data)):
                y = self.validate_data[i].dot(self.w)
                if y > 0:
                    prediction = 1
                else :
                    prediction = -1
                if prediction == self.validate_label[i] :
                    correct_num += 1
            accuracy = (correct_num / len(self.validate_data)) * 100 # 输出正确率百分比
            print("此次迭代的有效正确率为 %s"%str(accuracy))
    
    def predict(self, test_data):
        filename = 'result.txt'
        f = open(filename, 'w')
        test_lable = np.zeros((n_test,1))
        for i in range(len(test_data)):
            y = test_data[i].dot(self.w)
            if y > 0:
                test_lable[i] = 1
            else :
                test_lable[i] = -1
            f.write(str(test_lable[i])+'\n')
        f.close()
        
if __name__ == '__main__':
    train_data, train_label, validate_data, validate_label = loadDataSet("./data/train")
    test_data= loadTestSet("./data/test")
    # print(train_data, train_label, validate_data, validate_label)
    # print(test_data)
    svm = SoftMarginSVM(train_data, train_label, validate_data, validate_label)
    svm.fit()
    svm.predict(test_data)
    