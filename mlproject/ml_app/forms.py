from django import forms

class DatasetUploadForm(forms.Form):
    file = forms.FileField()