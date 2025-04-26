class InsufficientArgumentsException(Exception):
    def __init__(self, field):
        self.field = field
        self.message = f'Required Argument {field} not provided.'
        super().__init__(self.message)