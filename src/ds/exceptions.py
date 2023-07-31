class IOErrorAfterRetries(Exception):
    """Exception raised for IOErrors after retries."""

    def __init__(self, max_attempts, data_path):
        self.max_attempts = max_attempts
        self.message = f"Could not open file {data_path} after {max_attempts} attempts."
        super().__init__(self.message)


class ErrorAfterRetries(Exception):
    """Exception raised after retries"""

    def __init__(self, max_attempts, message):
        self.max_attempts = max_attempts
        self.message = f"{message} {max_attempts} failed attempts."
        super().__init__(self.message)
