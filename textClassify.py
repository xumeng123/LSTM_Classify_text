import tensorflow as tf
import numpy as np
rnn_unit = 200
input_size = 300
output_size = 2
batch_size = 100
time_step = 600
#max_length
testWordVecor = ''  #测试数据的vector向量
model_path=''    #模型所在路径
epoches = 5
lr = 0.001
dataLen = 25000  #数据长度，即句子的个数
testNum = 1000   #药测试数据的个数，不能超过25000
from tensorflow.contrib import rnn
#import data_process
import pickle


def getTextData(num):
    if num > dataLen:
        print('数据长度最大不超过25000,请重新输入testNum的值，此次退出！')
        exit()
    print('加载测试数据..........')
    file = open(testWordVecor,'rb')
    start = np.random.randint(0,dataLen - num-1)
    end = start + num;
    data = []
    label = []
    for i in range(start,end):
        t_data,t_label = pickle.load(file)
        data.append(t_data)
        label.append(t_label)
    file.close()
    data = np.array(data)
    label = np.array(label)
    print('测试数据加载完成！')
    return data,label,num




weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
       }


def LstmCell():
    lstm_cell = rnn.BasicLSTMCell(rnn_unit, state_is_tuple=True)
    #lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,output_keep_prob=0.75)
    return lstm_cell
def lstm(X,test=False):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入

    #cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell = rnn.MultiRNNCell([LstmCell() for _ in range(2)])
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    #output_rnn, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    #output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果

    output = tf.transpose(output_rnn,[1,0,2])
    #output = output[:,-1,:]
    output = output[-1]
    output=tf.reshape(output,[-1,rnn_unit]) #作为输出层的输入
    #output = tf.gather(output,int(output.get_shape()[0])-1)
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states,output



def text_classify(time_step=time_step):
    #print("batch_size: ",batch_size,'\n',"time_step: ",time_step,'\n',"hiden_unit: ",rnn_unit,'\n','input_size: ',input_size,'\n','output_size: ',output_size,'\n','train_step: ',train_step,'\n','learning_reate: ',lr)
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,output_size])
    pred,_,output=lstm(X)
    #corPred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    #accuracy = tf.reduce_mean(tf.cast(corPred,tf.float32))
    data,label,_ = getTextData(testNum)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('加载现有模型...')
        else:
            print("模型不存在，程序退出！！！")
            exit()
        f_d = {X:data}
        pred = sess.run(pred, feed_dict=f_d)
        pred = np.array(pred)
        corPred = np.equal(np.argmax(pred, 1), np.argmax(label, 1))
        accuracy = np.mean(np.array(corPred).astype(np.float32))
        print('测试结果为：\n')
        print('accuracy: ', accuracy)


text_classify()
