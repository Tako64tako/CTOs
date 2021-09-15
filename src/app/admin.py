from django.contrib import admin

from .models import Area
from .models import Cafe
from .models import Utility

admin.site.register(Area)
admin.site.register(Cafe)
admin.site.register(Utility)