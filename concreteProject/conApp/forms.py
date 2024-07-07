from django import forms
from .models import *


class conForm(forms.ModelForm):
    class Meta():
        model=conModel
        fields=['cement','slag','flyash','water','superplasticizer','coarseaggreate','fineaggregate','age']
