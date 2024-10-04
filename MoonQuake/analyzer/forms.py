from django import forms

class FileForm(forms.Form):
    file = forms.FileField(
        required=True,
        label='Upload Seismic Data',
    )
