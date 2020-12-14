import numpy as np
from tensorflow import keras
from keras import backend as K


def cosine_decay_with_warmup(global_step,learning_rate_base,total_steps,warmup_learning_rate=0.0,warmup_steps=0,hold_base_rate_steps=0):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')
        #The principle of cosine annealing is realized here, and the minimum value of learning rate is set as 0, so the expression is simplified
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *(global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #If hold_base_rate_steps are greater than 0, it indicates that the learning rate remains unchanged for a certain number of steps after warm up
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to ''warmup_learning_rate.')
        #The realization of linear growth
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #Only if global_step is still in the warmup stage will a linearly increased learning rate warmup_rate be used, otherwise a cosine annealing learning rate learning_rate will be used
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    Callback is inherited to realize the scheduling of learning rate
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        #learning_rates are used to record the learning rate after each update for graphical observation
        self.learning_rates = []
	#Update the Global Step and record the current learning rate
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
	#Renewal learning rate
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))