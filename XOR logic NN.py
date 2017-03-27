import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.figure(figsize = (10,8))
#fig = plt.figure(figsize=(10,7))

# 2-Dimensional input
a = np.array([0,0,1,1],dtype='float32')
b = np.array([0,1,0,1],dtype='float32')
c = np.array([0,1,1,0],dtype='float32').T
aa = np.concatenate([[a],[b]],axis=0).T

# 3-Dimensional input
'''
a = np.array([0,0,0,0,1,1,1,1],dtype='float32')
b = np.array([0,1,0,1,0,1,0,1],dtype='float32')
c = np.array([0,0,1,1,0,0,1,1],dtype='float32')
d = np.array([0,1,1,1,1,1,1,0],dtype='float32')

aa = np.concatenate([[a],[b],[c]],axis=0).T
'''
# 4-Dimensional input (requires an additional 3rd hidden unit to work)
'''
a = np.array([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],dtype='float32')
b = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],dtype='float32')
c = np.array([0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],dtype='float32')
c = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],dtype='float32')

e = np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],dtype='float32')

aa = np.concatenate([[a],[b],[c],[d]],axis=0).T
'''


height = aa.shape[0] # 8
width = aa.shape[1] # 3


x_input = tf.placeholder(tf.float32, [height,width])

x_in = tf.unstack(x_input)


n1 = 2
n2 = 1

w1 = tf.Variable(np.random.rand(width,n1),dtype=tf.float32)
w2 = tf.Variable(np.random.rand(n1,n2),dtype=tf.float32)

b1 = tf.Variable(np.random.rand(n1),dtype=tf.float32)
b2 = tf.Variable(np.random.rand(n2),dtype=tf.float32)

l1 = tf.nn.sigmoid( tf.nn.bias_add( tf.matmul(x_in, w1),b1))
l2 = tf.nn.sigmoid( tf.nn.bias_add( tf.matmul( l1 , w2),b2))

l2 = tf.transpose(l2)

print(l2.get_shape())




#loss =  tf.reduce_sum( tf.square( l2 - d )) # c, e, d depending on dimension
loss =  tf.reduce_sum( tf.square( l2 - c ))

var_list = [w1,b1, w2, b2]


train_step1 = tf.train.GradientDescentOptimizer(0.1).compute_gradients(loss, var_list)


# STRUCTURE: [ #variables, (0,1) tuple w 0 is grad 1 is weight, [tensor dim] ]

train_step2 = tf.train.GradientDescentOptimizer(0.1).apply_gradients(train_step1)


#var_grad = tf.gradients(loss, w2)[0]
#print(var_grad.get_shape())

train_steps = range(50000)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    w1s = np.array([np.zeros(width*n1)])
    b1s = np.array([[0,0]])
    w2s = np.array([[0,0]])
    b2s = np.array([[0]])
    
    for i in train_steps:

        
        _train_step1,_train_step2, _loss, _out = sess.run(
            [train_step1, train_step2,loss, l2],
            feed_dict = { x_input:aa})
        loss_list.append(_loss)
        
        if i%500==0:
            _out = np.transpose(_out)
            
            w1a = np.reshape(_train_step1[0][1][:],[-1])
            b1a = np.reshape(_train_step1[1][1][:],[-1])
            w2a = np.reshape(_train_step1[2][1][:],[-1])
            b2a = np.reshape(_train_step1[3][1][:],[-1])
            print(w1a,'w1a')
            '''
            print(w1a,'w1a')
            print(b1a,'b1a')
            print(w2a,'w2a')
            print(b2a,'b2a')

            print(w1s.shape, b1s.shape, w2s.shape, b2s.shape)
            '''

            print(len(w2s))
            #print(w2s)

            w1s = np.append(w1s, [w1a],axis=0) # 2x2
            b1s = np.append(b1s, [b1a],axis=0) # 2x1
            w2s = np.append(w2s, [w2a],axis=0) # 2x1
            b2s = np.append(b2s, b2a) # 1x1
            #print(_out)


            plt.subplot(2,2,1)
            plt.plot(a[[1,2]], b[[1,2]], 'ro',color='red',label='XOR 1 approx. Round: '+ str(i))
            plt.plot(a[[0,3]], b[[0,3]], 'ro',color='blue',label='XOR 0')
            plt.legend()

            plt.plot(_out[[0,1]], _out[[2,3]], 'ro',color='pink')
            plt.plot(_out[[0,1]], _out[[3,2]], 'ro',color='cyan')
            
            plt.subplot(2,2,2)
            plt.ylim(0,1)
            plt.text(0,0.1,c.T,fontsize=15)
            plt.text(0,0.3,_out,fontsize=15)
            plt.plot(loss_list,color='red',label='Loss')
            plt.legend()

            plt.subplot(2,2,3)
            #plt.ylim(-2,2)
            plt.plot(w1s[:,0],color='red',label='W11')
            plt.plot(w1s[:,1],color='cyan',label='W12')
            plt.plot(w1s[:,2],color='blue',label='W21')
            plt.plot(w1s[:,3],color='pink',label='W22')
            plt.plot(b1s[:,0],color='purple',label='b11')
            plt.plot(b1s[:,1],color='magenta',label='b12')
            plt.legend()

            plt.subplot(2,2,4)
            #plt.ylim(-2,2)
            plt.plot(w2s[:,0],color='red',label='W21')
            plt.plot(w2s[:,1],color='cyan',label='W22')
            plt.plot(b2s[:],color='yellow',label='b2')
            plt.legend()
            

            #plt.savefig(directory+'xor_graph'+str(i//50))



            
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            
