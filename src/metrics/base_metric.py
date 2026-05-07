from abc import abstractmethod


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    def __call__(self, **batch):
        """
        Convenience wrapper preserving the previous callable interface.
        """
        self.update(**batch)
        return self.compute()

    @abstractmethod
    def reset(self):
        """
        Reset the internal metric state before a new accumulation phase.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, **batch):
        """
        Update metric state from a new batch.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute(self):
        """
        Compute the final scalar value from the accumulated state.
        """
        raise NotImplementedError()
