class UserError(Exception):
    """A problem caused by the request (bad input, missing optional feature). Shown to the user as-is."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status
