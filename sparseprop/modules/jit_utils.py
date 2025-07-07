class JITOptions:
    def __init__(
        self, batch=0, do_unroll=False, do_parallel=False, do_only_scalar=False
    ):
        assert batch % 8 == 0, "Input sizes not multiples of 8 are unsupported!"
        assert not (do_only_scalar and do_unroll), (
            "Only Scalar mode in not yet supported for unrolling."
        )
        self.batch_size = batch
        self.unroll = do_unroll
        self.parallel = do_parallel
        self.unrolled_scalar = do_only_scalar
