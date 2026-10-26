class ApiError(Exception):
    """Erro de negócio da API, renderizado como {code, message, details}."""

    def __init__(self, status: int, code: str, message: str, details: dict | list | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details
