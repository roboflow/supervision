from __future__ import annotations


class IdCounter:
    def __init__(self, start_id: int = 0):
        """
        Initialize the ID counter.

        Args:
            start_id: The starting integer for the counter.

        Raises:
            ValueError: If start_id is less than or equal to -1.
        """
        self.start_id = start_id
        if self.start_id <= self.NO_ID:
            raise ValueError(f"start_id must be greater than {self.NO_ID}")
        self.reset()

    def reset(self) -> None:
        """Reset the counter to the initial start_id."""
        self._id = self.start_id

    def new_id(self) -> int:
        """
        Get the current ID and increment the counter.

        Returns:
            The newly assigned ID.
        """
        returned_id = self._id
        self._id += 1
        return returned_id

    @property
    def NO_ID(self) -> int:
        return -1
