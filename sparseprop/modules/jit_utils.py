class JITOptions:
    def __init__(self, batch=0, do_unroll=False, do_parallel=False):
        assert batch % 8 == 0, "Input sizes not multiples of 8 are unsupported!"

        self.batch_size = batch
        self.unroll = do_unroll
        self.parallel = do_parallel