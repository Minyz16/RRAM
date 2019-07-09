import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np


class RRAM:
    def __init__(self, M, N, W, R, num_state=3, non_drift_var=.001,
                 p_a_layers=(16,),
                 p_v_layers=(16,),
                 q_v_layers=(32, 16,),
                 learning_rate = .01,
                 gradient_clipping_norm=100.0,
                 train=True,
                 reuse = False
                 ):
        # write M times, N RRAM units
        # W: Mx1, R: MxN
        self.M = M
        self.N = N
        self.W = W
        self.R = R
        self.num_state = num_state
        self.non_drift_var = non_drift_var
        self.p_a_layers = p_a_layers
        self.p_v_layers = p_v_layers
        self.q_v_layers = q_v_layers
        self.train = train
        self.learning_rate = learning_rate
        self.gradient_clipping_norm = gradient_clipping_norm
        self.summaries = []
        with tf.variable_scope('RRAM', reuse=reuse):
            self.global_step = tf.get_variable(name="global_step", shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self._create_model()


    def _create_model(self):
        W = tf.expand_dims(self.W, 1)
        W = tf.tile(W, [1,self.N,1])   # W: MxNx1
        R = tf.expand_dims(self.R, -1)  # R: MxNx1

        with tf.variable_scope('variational_v'):    # q(V|R,W,theta)
            output = tf.concat([W,R], 2)    # [W R]: MxNx2
            for i in range(len(self.q_v_layers)):
                with tf.variable_scope('q_v_layers_' + str(i)) as scope:
                    output = layers.fully_connected(output, self.q_v_layers[i], scope=scope)
            with tf.variable_scope('q_v_layers_mean') as scope:
                self.v_mean = layers.fully_connected(output, 1, activation_fn=None, scope=scope)     # MxNx1
            with tf.variable_scope('q_v_layers_logvar') as scope:
                v_logvar = layers.fully_connected(output, 1, activation_fn=None, scope=scope)   # MxNx1
                self.v_var = tf.exp(v_logvar)
            v_sample = self.v_mean + tf.sqrt(self.v_var) * tf.random.normal([self.M, self.N, 1])    # MxNx1
            q_v = RRAM._gaussian_pdf(v_sample, self.v_mean, self.v_var)     # MxNx1

        with tf.variable_scope('forward_a'):
            output = W
            for i in range(len(self.p_a_layers)):
                with tf.variable_scope('p_a_layers_'+str(i)) as scope:
                    output = layers.fully_connected(output, self.p_a_layers[i], scope=scope)
            with tf.variable_scope('p_a_layers_' + str(len(self.p_a_layers))) as scope:
                output = layers.fully_connected(output, self.num_state, activation_fn=None, scope=scope)
                self.f_pi = tf.nn.softmax(output)          # f_pi: MxNx3 simplex

        with tf.variable_scope('forward_v'):
            output = W
            for i in range(len(self.p_v_layers)):
                with tf.variable_scope('p_v_layers_'+str(i)) as scope:
                    output = layers.fully_connected(output, self.p_v_layers[i], scope=scope)
            with tf.variable_scope('p_v_layers_lam1') as scope:
                loglam1 = layers.fully_connected(output, 1, activation_fn=None, scope=scope)
                self.lam1 = tf.exp(loglam1)
            with tf.variable_scope('p_v_layers_lam2') as scope:
                loglam2 = layers.fully_connected(output, 1, activation_fn=None, scope=scope)
                self.lam2 = tf.exp(loglam2)                 # lam1, lam2: MxNx1 positive number

        with tf.variable_scope('forward_r'):
            self.r_logvar = tf.get_variable('r_logvar', [])
            p_r = RRAM._gaussian_pdf(R, v_sample, tf.exp(self.r_logvar))    # MxNx1

        with tf.variable_scope('ELBO'):
            p_list = []
            for i in range(3):
                a = tf.constant(i, shape=[self.M, self.N, 1])
                p_a = RRAM._categorical_pdf(a, self.f_pi)
                p_v = RRAM._exponential_pdf(v_sample, a, W, self.lam1, self.lam2, self.non_drift_var)
                p_list.append(p_a * p_v)
            eps = 10e-10
            ELBO = tf.log(tf.add_n(p_list) + eps) + tf.log(p_r + eps) - tf.log(q_v + eps)

            self.ELBO = tf.reduce_sum(ELBO)

        if self.train:
            with tf.variable_scope('training'):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                grads, variables = zip(*optimizer.compute_gradients(-self.ELBO))
                if self.gradient_clipping_norm is not None:
                    grads = tf.clip_by_global_norm(grads, self.gradient_clipping_norm)[0]
                grads_and_vars = list(zip(grads, variables))
                self.training = optimizer.apply_gradients(
                    grads_and_vars, global_step=self.global_step
                )

        with tf.variable_scope('summaries'):
            self.summaries.append(tf.summary.scalar('ELBO', self.ELBO/self.M))


    @staticmethod
    def _gaussian_pdf(x, mean, var):
        # x, mean, var: MxNx1, return: MxNx1
        return tf.exp(- tf.square(x - mean) / (2 * var)) / tf.sqrt(2 * np.pi * var)


    @staticmethod
    def _exponential_pdf(x, state, y, lam1, lam2, non_drift_var):
        # x, y: MxNx1, state: MxNx1 with element in [0,1,2], lam1, lam2: MxNx1 positive number
        M = tf.shape(x)[0]
        N = tf.shape(x)[1]
        x = tf.reshape(x, [M * N, 1])
        y = tf.reshape(y, [M * N, 1])
        state = tf.reshape(state, [M * N, 1])
        lam1 = tf.reshape(lam1, [M * N, 1])
        lam2 = tf.reshape(lam2, [M * N, 1])
        elems = tf.concat([tf.cast(state, dtype=tf.float32), x, y, lam1, lam2], 1)     # MNx5
        def fn(xx):
            s_state = tf.cast(xx[0], tf.int32)
            s_x = xx[1]
            s_y = xx[2]
            l1 = xx[3]
            l2 = xx[4]
            # state 0, positive drift
            def c0():
                return tf.cond(s_x >= s_y, lambda: l1 * tf.exp(-l1 * (s_x - s_y)), lambda: tf.constant(0.))
            # state 1, negative drift
            def c1():
                return tf.cond(s_y >= s_x, lambda: l2 * tf.exp(-l2 * (s_y - s_x)), lambda: tf.constant(0.))
            # state 2, non-drift
            def c2():
                return RRAM._gaussian_pdf(s_x, s_y, non_drift_var)
            return tf.case({tf.equal(s_state, 0):c0,
                tf.equal(s_state, 1): c1
            }, default=c2, exclusive=True)
        tmp = tf.map_fn(fn=fn, elems=elems, dtype=tf.float32)
        return tf.reshape(tmp, [M, N, 1])


    @staticmethod
    def _categorical_pdf(x, prob):
        # x: MxNx1 with element in [0,1,2], prob: MxNx3 simplex
        elems = tf.concat([tf.cast(x, dtype=tf.float32),prob], 2)      # MxNx4
        M = tf.shape(elems)[0]
        N = tf.shape(elems)[1]
        elems = tf.reshape(elems, [M * N, 4])

        def fn(y):
            index = tf.cast(y[0], tf.int32)
            params = y[1:]
            return params[index]

        tmp = tf.map_fn(fn=fn, elems=elems, dtype=tf.float32)
        return tf.reshape(tmp, [M, N, 1])





