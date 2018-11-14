from skimage import io, transform
import tensorflow as tf
import numpy as np


path1 = "E:/data/datasets/dieases_photos/apple2_ban/apple16.jpg"
path2 = "E:/data/datasets/dieases_photos/apple_lun/apple1.jpg"


flower_dict = {0: '斑点落叶病', 1: '红蜘蛛', 2: '炭除病', 3: '轮纹病'}

w = 100
h = 100
c = 3


def addElementToDict(element):
    flower_dict.update(element)
    return flower_dict


def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img, (w, h))
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data.append(data1)
    data.append(data2)


    saver = tf.train.import_meta_graph('E:/data/model/dieases/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('E:/data/model/dieases/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第", i+1, "朵花预测:"+flower_dict[output[i]])
