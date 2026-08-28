from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """Modelo de usuário próprio — sem campos extras ainda, mas trocável desde o dia 1."""

    pass
