class JITOptions:
    def __init__(
        self,
        batch=0,
        do_unroll=False,
        do_parallel=False,
        do_only_scalar=False,
        do_psc=False,
        do_reg_tiling=False,
        reg_tile_size=(),
    ):
        assert batch % 8 == 0, "Input sizes not multiples of 8 are unsupported!"
        assert not (do_only_scalar and do_unroll), (
            "Only Scalar mode in not yet supported for unrolling."
        )
        assert not (do_reg_tiling ^ bool(reg_tile_size)), (
            "If one register tiling arg is given, the other must be too."
        )
        assert (do_psc + do_reg_tiling + do_only_scalar) <= 1, (
            "Usage of any of these two (or more versions) is not supported."
        )
        self.batch_size = batch
        self.unroll = do_unroll
        self.parallel = do_parallel
        self.unrolled_scalar = do_only_scalar
        self.psc = do_psc
        self.reg_tiling = do_reg_tiling
        self.reg_tile_size = reg_tile_size
