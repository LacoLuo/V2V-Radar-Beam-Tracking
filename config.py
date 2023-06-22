class configurations(object):
    def __init__(self):
        # Dataset param.
        self.len_of_sequence = 10

        # Train param.
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_steps = 100000
        self.summary_steps = 100
        self.store_steps = 100
        self.gpu = 1
