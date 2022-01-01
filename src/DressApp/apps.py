from django.apps import AppConfig


class DressappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'DressApp'

    def ready(self):
        from . import signals
