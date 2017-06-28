import tensorflow as tf

class PatchCNN:
    def __init__(self, CNNConfig):
        if CNNConfig["train_flag"]:
            self.patch = tf.placeholder("float32", [None, CNNConfig["patch_size"], CNNConfig["patch_size"], 3])
            self.patch_t = tf.placeholder("float32", [None, CNNConfig["patch_size"], CNNConfig["patch_size"], 3])
            self.label = tf.placeholder("float32", [None, CNNConfig["descriptor_dim"]])
        else:
            self.patch = tf.placeholder("float32", [1, None, None, 3])

        self.alpha = CNNConfig["alpha"]
        self.descriptor_dim = CNNConfig["descriptor_dim"]
        self._patch_size = CNNConfig["patch_size"]

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.model(self.patch)
            scope.reuse_variables()
            self.o2 = self.model(self.patch_t)
            
        self.o1_flat = tf.reshape(self.o1, [-1, self.descriptor_dim])
        self.o2_flat = tf.reshape(self.o2, [-1, self.descriptor_dim])

        self.cost, self.inver_loss, self.covariance_loss \
                = self.regression_loss()

    def weight_variable(self, name, shape):
        weight = tf.get_variable(name = name+'_W', shape = shape, initializer = tf.random_normal_initializer(0, 1.0))
        return weight

    def bias_variable(self,name, shape):
        bias = tf.get_variable(name = name + '_b', shape = shape, initializer = tf.constant_initializer(0.0))
        return bias

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def conv2d_layer(self, name, shape, x):
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.relu(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias)
        return conv_val

    def conv2d_layer_no_relu(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias
        return conv_val

    def fc_layer(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[1]], wd = 1e-5);

        fc_val = tf.matmul(x, weight)+bias
        return fc_val

    def conv2d_layer_BN(self, name, shape, x):
        weight = self._variable_with_weight_decay(name=name + '_W', shape = shape, wd = 1e-5);
        bias = self._variable_with_weight_decay(name=name + '_b', shape = [shape[3]], wd = 1e-5);

        conv_val = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='VALID')+bias

        batch_mean2, batch_var2 = tf.nn.moments(conv_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[3]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[3]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(conv_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        
        return tf.nn.relu(bn_val)

    def fc_layer_BN(self, name, shape, x):
        weight_init = tf.truncated_normal_initializer(stddev=1.0)
        weight = tf.get_variable(name=name + '_W', dtype = tf.float32, shape=shape, initializer = weight_init)
        bias = tf.get_variable(name=name + '_b', dtype = tf.float32,\
                               initializer = tf.constant(0.0, shape = [shape[1]], dtype = tf.float32))

        fc_val = tf.matmul(x, weight)+bias
        batch_mean2, batch_var2 = tf.nn.moments(fc_val,[0])
        scale2 = tf.get_variable(name = name+ '_bn_w', dtype = tf.float32, \
                               initializer = tf.constant(1, shape = [shape[1]], dtype = tf.float32))
        beta2 = tf.get_variable(name = name+ '_bn_b', dtype = tf.float32, \
                               initializer = tf.constant(0, shape = [shape[1]], dtype = tf.float32))
        bn_val = tf.nn.batch_normalization(fc_val,batch_mean2,batch_var2,beta2,scale2,1e-3)
        return bn_val

    def _variable_with_weight_decay(self, name, shape, wd):
        dtype = tf.float32
        weight_init = tf.uniform_unit_scaling_initializer(factor=0.5)
        #weight_init = tf.truncated_normal_initializer(stddev=1.0)
        var = tf.get_variable(name=name, dtype = tf.float32, \
                shape=shape, initializer = weight_init)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    def model(self, x):
        if self._patch_size==16:
            h_conv1 = self.conv2d_layer_BN('conv1', [5, 5, 3, 32], x)
            h_conv2 = self.conv2d_layer_BN('conv2', [5, 5, 32, 32], h_conv1)
            h_conv3 = self.conv2d_layer_BN('conv3', [5, 5, 32, 64], h_conv2)
            h_conv4 = self.conv2d_layer_BN('conv4', [3, 3, 64, 64], h_conv3)
            h_conv5 = self.conv2d_layer_BN('conv5', [2, 2, 64, 128], h_conv4)
            conv5_flatten = tf.reshape(h_conv5, [-1, 128])
            output = self.fc_layer_BN('fc1',[128,self.descriptor_dim],conv5_flatten)

        elif self._patch_size == 32:
            h_conv1 = self.conv2d_layer_BN('conv1', [5, 5, 3, 32], x)
            h_pool1 = self.max_pool_2x2(h_conv1)
            h_conv2 = self.conv2d_layer_BN('conv2', [5, 5, 32, 128], h_pool1)
            h_pool2 = self.max_pool_2x2(h_conv2)
            h_conv3 = self.conv2d_layer_BN('conv3', [3, 3, 128, 128], h_pool2)
            h_conv4 = self.conv2d_layer_BN('conv4', [3, 3, 128, 256], h_conv3)
            output = self.conv2d_layer_no_relu('fc1',[1, 1, 256, self.descriptor_dim],h_conv4)
 
        elif self._patch_size == 64:
            h_conv1 = self.conv2d_layer_BN('conv1', [7, 7, 1, 32], x)
            h_pool1 = self.max_pool_2x2(h_conv1)
            h_conv2 = self.conv2d_layer_BN('conv2', [6, 6, 32, 64], h_pool1)
            h_pool2 = self.max_pool_2x2(h_conv2)
            h_conv3 = self.conv2d_layer_BN('conv3', [5, 5, 64, 128], h_pool2)
            h_pool3 = self.max_pool_2x2(h_conv3)
            pool3_flatten = tf.reshape(h_pool3, [-1, 4*4*128])
            output = self.fc_layer_BN('fc1',[4*4*128,self.descriptor_dim],pool3_flatten)
        else:
            output = []

        return  output

    #regression loss with invertable loss to standard patches
    def regression_loss(self):
        alpha_tf = tf.constant(self.alpha)
        with tf.name_scope('all_loss'):
            #invertable loss for standard patches
            with tf.name_scope('inver_loss'):
                inver_loss = tf.reduce_mean(tf.reduce_mean(tf.pow(self.o1_flat,2),1))
            #covariance loss for transformed patches
            with tf.name_scope('covariance_loss'):
                covariance_loss = tf.reduce_mean(tf.reduce_mean(tf.pow(tf.subtract(self.o2_flat,tf.add(self.o1_flat,self.label)),2),1))
            #total loss
            with tf.name_scope('loss'):
                loss = tf.multiply(alpha_tf,inver_loss) + covariance_loss
        #write summary 
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('inver_loss', inver_loss)
        tf.summary.scalar('covariance_loss', covariance_loss)
        
        return loss, inver_loss, covariance_loss
