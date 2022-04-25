class Metric():
    def __init__(self):
        self._values = []

    def _calculate_metric(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0

    def update(self, current_value) -> None:
        self._values.append(current_value)

    def read(self):
        return self._calculate_metric()

    def reset(self):
        self._values = []
