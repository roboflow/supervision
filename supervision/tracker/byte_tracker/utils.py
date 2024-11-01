class IdCounter:
    def __init__(self, start_id: int = 0):
        self.start_id = start_id
        if self.start_id <= self.NO_ID:
            raise ValueError(f"start_id must be greater than {self.NO_ID}")
        self.reset()

    def reset(self) -> None:
        self._id = self.start_id

    def new_id(self) -> int:
        returned_id = self._id
        self._id += 1
        return returned_id

    @property
    def NO_ID(self) -> int:
        return -1
