from django.db import models

# Create your models here.
class UploadVideos(models.Model):
    video = models.FileField(upload_to="./static/media/")
    class Meta:
        db_table = "videos"