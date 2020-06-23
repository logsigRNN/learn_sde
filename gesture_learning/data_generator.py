import keras.utils
import math
import copy
import random
import numpy as np

def rxf(a):
    x = np.array([1, 0, 0, 0,
                 0, np.cos(a), np.sin(a), 0,
                 0, -np.sin(a), np.cos(a), 0,
                 0, 0, 0, 1])
    return x.reshape(4,4)

def ryf(b):
    y = np.array([np.cos(b), 0, -np.sin(b), 0,
                 0, 1, 0, 0,
                 np.sin(b), 0, np.cos(b), 0,
                 0, 0, 0, 1])
    return y.reshape(4,4)

def rzf(c):
    z = np.array([np.cos(c), np.sin(c), 0, 0,
                 -np.sin(c), np.cos(c), 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1])
    return z.reshape(4,4)

"""
    A func to do rotation augmentation
    Here matrix is a (J, 3) array, J is the number of joints, 3 means x, y, z three dimension.
"""
def rotation(matrix, a, b, c):
    rx = rxf(a)
    ry = ryf(b)
    rz = rzf(c)

    nums, _ = matrix.shape
    newmatrix = []

    xyzone = np.ones((1,4))

    for i in range(nums):
        xyzone[0][:3] = matrix[i]
        xyznew = np.dot(np.dot(np.dot((rx.T), (ry.T)), (rz.T)), (xyzone.T))
        xyznew = xyznew.reshape(4)
        newmatrix.append(xyznew[0:3])
    matrix = np.array(newmatrix)

    return matrix

def translation(src_samples, trans_nb_max, final_frame_nb):
    assert (trans_nb_max>=0) and (trans_nb_max<=final_frame_nb/5)
    num, dim = src_samples.shape
    # print(src_samples.shape)
    dst_samples = np.zeros(src_samples.shape)
    trans_nb_list = [random.sample([i for i in range(-trans_nb_max, trans_nb_max)], 1)[0] \
        for _ in range(num)]

    for k in range(num):
        trans_nb = trans_nb_list[k]
        if trans_nb==0:
            dst_samples[k,:] = src_samples[k,:]
        elif trans_nb<0:
            src_arr = src_samples[k,:]
            dst_arr_l = np.zeros(dim)
            trans_nb = -trans_nb
            for i in range(dim//final_frame_nb):
                # left shift.
                dst_arr_l[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb] = \
                    src_arr[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb]
                dst_arr_l[(i+1)*final_frame_nb-trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[(i+1)*final_frame_nb-1]
            dst_samples[k,:] = dst_arr_l
        else:
            src_arr = src_samples[k,:]
            dst_arr_r = np.zeros(dim)
            for i in range(dim//final_frame_nb):
                # right shift.
                dst_arr_r[i*final_frame_nb+trans_nb:(i+1)*final_frame_nb] = \
                    src_arr[i*final_frame_nb:(i+1)*final_frame_nb-trans_nb]
                dst_arr_r[i*final_frame_nb:i*final_frame_nb+trans_nb] = \
                    src_arr[i*final_frame_nb]
            dst_samples[k,:] = dst_arr_r
    return dst_samples




class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, labels, batch_size=40, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.labels = labels
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]
        batch_labels = [self.labels[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas, batch_labels)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas, batch_labels):
        images = []
        labels = []
        batch_datas = np.array(batch_datas).reshape(-1,39,57)
        aug_train = copy.deepcopy(batch_datas)
        aug_train_label = copy.deepcopy(batch_labels)
        tmp_size = len(batch_datas)
        n_joints = batch_datas[0].shape[1]
        # 生成数据
        for i in range(tmp_size):
            a = np.random.uniform(-np.pi/36,np.pi/36)
            b = np.random.uniform(-np.pi/18,np.pi/18)
            c = np.random.uniform(-np.pi/36,np.pi/36)
            tmpsample=np.zeros((1,39,n_joints))         
            for j in range(39):
                tmpmat = batch_datas[i][j].reshape(int(n_joints/3),3)
                tmpmat = rotation(tmpmat, a,b,c)
                tmpsample[0][j] = tmpmat.reshape(n_joints)
            aug_train = np.concatenate((aug_train, tmpsample), axis=0)
        aug_train_label = np.concatenate((aug_train_label, batch_labels),axis=0)
        
        tmp_train = np.zeros((tmp_size,39*n_joints))
        for i in range(tmp_size):
            tmp_train[i] = batch_datas[i].T.reshape(39*n_joints)
        tmp_train = translation(tmp_train, 5, 39)
        tmp_train2 = np.zeros((tmp_size, 39,n_joints))
        for i in range(tmp_size):
            tmp_train2[i] = tmp_train[i].reshape(n_joints,39).T
        aug_train = np.concatenate((aug_train, tmp_train2),axis=0)
        aug_train_label = np.concatenate((aug_train_label, batch_labels), axis=0)
        
        tmp_train3 = np.random.normal(0,0.001,(tmp_size,39,n_joints))+batch_datas
        aug_train = np.concatenate((aug_train, tmp_train3),axis=0)
        aug_train_label = np.concatenate((aug_train_label, batch_labels), axis=0)

        aug_train = aug_train.reshape(-1,39,19,3)

        return aug_train, aug_train_label

