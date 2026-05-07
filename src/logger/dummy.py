class DummyWriter:
    """
    No-op experiment writer for local/debug training without W&B or Comet.
    """

    def __init__(
        self,
        logger=None,
        project_config=None,
        run_name="debug",
        loss_names=None,
        log_checkpoints=False,
        id_length=8,
        **kwargs,
    ):
        self.run_name = run_name
        self.loss_names = loss_names or ["loss"]
        self.log_checkpoints = log_checkpoints
        self.id_length = id_length
        self.run_id = None
        self.step = 0
        self.mode = "train"

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_checkpoint(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_audio(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_table(self, *args, **kwargs):
        pass
