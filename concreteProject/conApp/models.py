from django.db import models

# Create your models here.
class conModel(models.Model):

    cement=models.FloatField()
    slag=models.FloatField()
    flyash=models.FloatField()
    water=models.FloatField()
    superplasticizer=models.FloatField()
    coarseaggreate=models.FloatField()
    fineaggregate=models.FloatField()
    age=models.FloatField()
