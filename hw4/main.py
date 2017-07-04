import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from tensorflow.contrib.distributions import Normal

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    #b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    b = tf.get_variable(name + "/b", [size])
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):

    def __init__(self, **kwargs):
        self.sess = tf.Session()

        layers = [32, 16, 1]
        in_size = kwargs.pop("ob_dim")
        in_size = self.preproc(np.zeros([1, in_size])).shape[1]
        self.layers = [in_size] + layers
        self.lr = kwargs.pop("stepsize")
        self.n_epochs = kwargs.pop("n_epochs")

        self.init_weight()
        self.add_placeholders()

        self.base_value = self.forward(self.ob_ph)
        self.loss = self.compute_loss(self.base_value)
        self.train_op = self.add_train_op(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

    def init_weight(self):
        n_hid = len(self.layers) - 1
        self.weights = {}
        self.biases = {}
        for i in range(n_hid):
            self.weights["weight/h%s"%i] = tf.get_variable("weight/h%s"%i, 
                    [self.layers[i], self.layers[i+1]], 
                    initializer=tf.random_normal_initializer(0., 0.1))
            self.biases["bias/h%s"%i] = tf.get_variable("bias/h%s"%i, 
                    [self.layers[i+1]], 
                    initializer=tf.random_uniform_initializer(-0.1, 0.1))

    def add_placeholders(self):
        #self.lr_ph = tf.placeholder(shape=[], name="lr", dtype=tf.float32)
        self.ob_ph = tf.placeholder(
                shape=[None, self.layers[0]], name="obs", dtype=tf.float32)
        self.return_ph = tf.placeholder(
                shape=[None], name="return", dtype=tf.float32)

    def forward(self, x):
        h = x
        n_layers = len(self.layers)
        for i in range(n_layers-2):
            h = tf.matmul(h, self.weights["weight/h%s"%i]) + self.biases["bias/h%s"%i]
            h = lrelu(h, 0.1)
        i = n_layers - 2
        h = tf.matmul(h, self.weights["weight/h%s"%i]) + self.biases["bias/h%s"%i]
        return h

    def compute_loss(self, b):
        loss = tf.nn.l2_loss(self.return_ph - b)
        return loss

    def add_train_op(self, loss):
        opt = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return opt

    def fit(self, X, y):
        Xp = self.preproc(X)
        for i in range(self.n_epochs):
            self.sess.run(self.train_op, 
                    feed_dict={self.ob_ph:Xp, 
                        self.return_ph:y})

    def predict(self, X):
        Xp = self.preproc(X)
        b = self.sess.run(self.base_value, 
                feed_dict={self.ob_ph:Xp})
        return b.squeeze()

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)



def main_cartpole(logdir, seed, n_iter, gamma, min_timesteps_per_batch, stepsize, vf_type, vf_params, animate=False):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    if vf_type == "linear":
        vf = LinearValueFunction()
    elif vf_type == "nn":
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)
    else:
        raise Exception("vf_type must be either 'linear' or 'nn'!")

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)


    ####
    # YOUR_CODE_HERE

    # batch of observations
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    # batch of actions 
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.float32)
    # batch of advantage function estimates
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    
    # 2-layer network to learn state from observation
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0)))
    sy_h2 = lrelu(dense(sy_h1, 32, "h2", weight_init=normc_initializer(1.0)))
    # Mean control output
    sy_mean_na = dense(sy_h2, ac_dim, "mean", weight_init=normc_initializer(0.1))
    # Variance
    logstd_a = tf.get_variable("logstdev", [ac_dim])

    # define action distribution
    sy_ac_distr = Normal(
            mu=tf.squeeze(sy_mean_na), 
            sigma=tf.exp(logstd_a),
            validate_args=True)
    # sampled actions, used for defining the policy 
    # (NOT computing the policy gradient)
    sy_sampled_ac = tf.squeeze(sy_ac_distr.sample(sample_shape=[ac_dim]))

    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = sy_ac_distr.log_pdf(sy_ac_n)


    # used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES
    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_oldlogstd_a = tf.placeholder(shape=[ac_dim], name="oldlogstdev", dtype=tf.float32)
    sy_ac_olddistr = Normal(
            mu=tf.squeeze(sy_oldmean_na), 
            sigma=tf.exp(sy_oldlogstd_a),
            validate_args=True)

    sy_kl = tf.reduce_mean(tf.contrib.distributions.kl(sy_ac_distr, sy_ac_olddistr))
    sy_ent = tf.reduce_mean(sy_ac_distr.entropy())

    ####

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    total_timesteps = 0
    stepsize = initial_stepsize

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        ####
        # YOUR_CODE_HERE

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step([ac])
                rewards.append(rew)
                if done:
                    break                    

            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break

        total_timesteps += timesteps_this_batch

        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldmean_na, oldlogstdev = sess.run(
                [update_op, sy_mean_na, logstd_a], 
                feed_dict={
                    sy_ob_no:ob_no, 
                    sy_ac_n:ac_n, 
                    sy_adv_n:standardized_adv_n, 
                    sy_stepsize:stepsize
                    })
        kl, ent = sess.run(
                [sy_kl, sy_ent], 
                feed_dict={
                    sy_ob_no:ob_no, 
                    sy_oldmean_na:oldmean_na, 
                    sy_oldlogstd_a:oldlogstdev
                    })


        ####

        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')


        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def main_pendulum1(d):
    return main_pendulum(**d)

if __name__ == "__main__":
    game = "pendulum"
    vf_type = "nn"
    logpath = "result/" + game + "_" + vf_type

    if game == "cartpole":
        cartpole_params = dict(
                logdir=logpath, 
                seed=0, 
                n_iter=100, 
                gamma=1.0, 
                min_timesteps_per_batch=1000, 
                stepsize=1e-2, 
                vf_type=vf_type, 
                vf_params={} if vf_type == "linear" else \
                        dict(n_epochs=10, stepsize=1e-3), 
                animate=True 
                )
        main_cartpole(**cartpole_params) # when you want to start collecting results, set the logdir
    elif game == "pendulum":
        pendulum_params = dict(
                logdir=logpath, 
                seed=0, 
                n_iter=400, 
                gamma=0.97, 
                min_timesteps_per_batch=2500, 
                initial_stepsize=1e-3, 
                desired_kl=2e-3, 
                vf_type=vf_type, 
                vf_params={} if vf_type == "linear" else \
                        dict(n_epochs=10, stepsize=1e-3), 
                animate=True 
                )
        main_pendulum(**pendulum_params)
    if 0:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
        params = [
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        import multiprocessing
        p = multiprocessing.Pool()
        p.map(main_pendulum1, params)
