import glob
import os
import tensorflow as tf
import numpy as np
import time
from skimage import io, transform

#数据集地址
path='E:/data/datasets/dieases_photos/'
#模型保存地址
model_path='E:/data/model/dieases/model.ckpt'

#将所有的图片resize成100*100
w = 100
h = 100
c = 3


#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]  #给文件夹排序号，0是0文件夹，1是1文件夹...
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'% (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h, c))
            imgs.append(img)
            labels.append(idx)
            print('reading the idx:%s' % (idx))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# 样本和标签的读入与分类
data, label = read_img(path)

#打乱顺序
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]


#将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example*ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

#-----------------构建CNN神经网络模型----------------------
#数据占位符
x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):  # 开启一个联系上下文的命名空间，空间名是layer1-conv1，在tf.get_variable可以顺利调用
        conv1_weights = tf.get_variable("weight", [5, 5, 3, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 上面一行命令是生成卷积核：是一个tansor类型，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
        # tf.truncated_normal_initializer：从截断的正态分布中输出随机值。这是神经网络权重和过滤器的推荐初始值。
        # mean：一个python标量或一个标量张量。要生成的随机值的均值。
        # stddev：一个python标量或一个标量张量。要生成的随机值的标准偏差。
        # seed：一个Python整数。用于创建随机种子。查看 tf.set_random_seed 行为。
        # dtype：数据类型。只支持浮点类型。

        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
        # 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
        # 具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

        # 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

        # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
        # 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
        # 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true#
        # 结果返回一个Tensor，这个输出，就是我们常说的feature map特征图，shape仍然是[batch, height, width, channels]这种形式。

        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        # 激活函数，非最大值置零
        # 这个函数的作用是计算激活函数 relu，即 max(features, 0)。即将矩阵中每行的非最大值置0。

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        # tf.nn.max_pool(value, ksize, strides, padding, name=None)
        # 参数是四个，和卷积很类似：
        # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
        # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
        # 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
        # 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式


    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # [5,5,32,64] 5表示本次卷积核高宽，32表示经过上一层32个卷积核的卷积，我们有了32张特征图，64表明本次会有64个卷积核卷积
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6*6*128
        reshaped = tf.reshape(pool4, [-1, nodes])
        # tf.reshape(tensor(矩阵),shape(维度),name=None)
        # 改变一个矩阵的维度，可以从多维变到一维，也可以从一维变到多维
        # 其中，-1参数表示不确定，可由函数自己计算出来，原矩阵/一个维度=另一个维度

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        # tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表
        # 在深度学习中，通常用这几个函数存放不同层中的权值和偏置参数，
        # 也就是把所有可学习参数利用tf.contrib.layers.l2_regularizer(regular_num)(w)得到norm后，都放到’regular’的列表中作为正则项，
        # 然后使用tf.add_n函数将他们和原本的loss相加，得到含有正则的loss。

        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases) # MCP模型
        # tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
        # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None,name=None)
        # 上面方法中常用的是前两个参数：
        # 第一个参数x：指输入的数据。
        # 第二个参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符,  keep_prob = tf.placeholder(tf.float32) 。
        # tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
        # 第五个参数name：指定该操作的名字

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 4],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [4], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


# 定义规则化方法，并计算网络激活值
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
#两种思想都是希望限制权重的大小，使得模型不能拟合训练数据中的随机噪点。(两种思想，就是两个公式，因为是图，就没贴出来)
#两种方式在TensorFlow中的提供的函数为：
#tf.contrib.layers.l1_regularizer(scale, scope=None) 其中scale为权值(这个权值会乘以w的值，MCP的内个w，江湖传闻w和过拟合值有说不清的关系)
#tf.contrib.layers.l2_regularizer(scale, scope=None)

#x是输入的图像的tansor，logits是经过卷积、池化、全连接处理处理过的数据
logits = inference(x, False, regularizer)



#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

#计算logits 和 labels 之间的稀疏softmax 交叉熵 这个是计算误差率
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
# tf.train.AdamOptimizer 优化器中的梯度优化函数（作用是依据learning_rate步长，来最小化loss误差率）
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，
#这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
# 求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 参数1--input_tensor:待求值的tensor。
# 参数2--reduction_indices:在哪一维上求解
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
#四个参数是：训练数据，测试数据，用户输入的每批训练的数据数量，shuffle是洗牌的意思，这里表示是否开始随机
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)#assert断言机制，如果后面的表达式为真，则直接抛出异常。在这里的意思,大概就是:样本和标签数量要对上
    if shuffle:
        # 生成一个np.arange可迭代长度是len(训练数据),也就是训练数据第一维数据的数量(就是训练数据的数量，训练图片的数量)
        indices = np.arange(len(inputs))
        # np.random.shuffle打乱arange中的顺序，使其随机循序化，如果是数组，只打乱第一维
        np.random.shuffle(indices)
    # 这个range(初始值为0，终止值为[训练图片数-每批训练图片数+1]，步长是[每批训练图片数])：例(0[起始值],80[训练图片数]-20[每批训练图片数],20[每批训练图片数]),也就是(0,60,20)当循环到60时,会加20到达80的训练样本
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            # 如果shuffle为真,将indices列表,切片(一批)赋值给excerpt
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        # yield常见用法：该关键字用于函数中会把函数包装为generator。然后可以对该generator进行迭代: for x in fun(param).
        # 按照我的理解，可以把yield的功效理解为暂停和播放。
        # 在一个函数中，程序执行到yield语句的时候，程序暂停，返回yield后面表达式的值，在下一次调用的时候，从yield语句暂停的地方继续执行，如此循环，直到函数执行完。
        # 此处,就是返回每次循环中 从inputs和targets列表中,截取的 经过上面slice()切片函数定义过的 数据.
        # (最后的shuffle变量，决定了样本是否随机化)


#训练和测试数据，可将n_epoch设置更大一些
n_epoch = 5
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
count = 0
# 训练多少遍，FLAGS.epoch是用户输入的，比如是10，也就是把样本遍历10遍
for epoch in range(n_epoch):
    start_time = time.time()
    count += 1
    ### 单次训练部分  此处for循环结束之日，就是训练样本遍历了一遍之时
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))

    ### 单次验证部分   具体和上面雷同，下面是计算的测试数据，不用梯度优化
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    print("   validation acc: %f" % (np.sum(val_acc) / n_batch))

    print("这是第 %d 次batch测试和验证" % count)
saver.save(sess, model_path)
sess.close()
