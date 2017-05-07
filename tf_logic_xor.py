import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
data1=np.random.randn(50,3)
data2=np.random.randn(50,3)+[5,5,0]
data1[:,2]=0
data2[:,2]=1
data=np.vstack((data1,data2))
np.random.shuffle(data)
data=np.array(data,dtype=np.float32)
print(data)
plt.scatter(data[data[:,2]==0][:,0],data[data[:,2]==0][:,1])
plt.scatter(data[data[:,2]==1][:,0],data[data[:,2]==1][:,1])
plt.show()
'''
data=np.array([[0,0,0], # [x,y,label]
               [1,0,1],
               [0,1,1],
               [1,1,0]],dtype=np.float32)
x=data[:,:2]
label=data[:,2].reshape(4,1) # y2's shape is [4,1]

W1=tf.Variable(tf.random_normal([2,2]))
b1=tf.Variable(tf.random_normal([2]))
W2=tf.Variable(tf.random_normal([2,1]))
#b2=tf.Variable(np.random.normal())
b2=tf.Variable(tf.random_normal([1]))

y1=tf.nn.sigmoid(tf.matmul(x,W1)+b1)
y2=tf.nn.sigmoid(tf.matmul(y1,W2)+b2)
cost=-tf.reduce_mean(label*tf.log(y2)+(1-label)*tf.log(1-y2))
train=tf.train.GradientDescentOptimizer(0.4).minimize(cost)
cost_list=[]
y_list=[]

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for batch in range(10001):
        sess.run(train)
        if batch%100==0:
            c=cost.eval()
            cost_list.append(c)
            y_list.append(list(y2.eval().ravel()))
            print(">",batch,c)
            if c<0.01: break

    print("> reuslt:",y2.eval())
    y_array=np.array(y_list)
    plt.plot(cost_list,">-",label="cost")
    plt.plot(y_array[:,0],"--",label="(0,0)")
    plt.plot(y_array[:,1],"--",label="(1,0)")
    plt.plot(y_array[:,2],"--",label="(0,1)")
    plt.plot(y_array[:,3],"--",label="(1,1)")
    plt.legend()
    plt.show()
  
