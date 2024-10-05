from django import forms

MODEL_CHOICES = [
    ("lunar", "lunar"),
    ("mars", "mars")
]
class FileForm(forms.Form):
    file = forms.FileField(
        required=True,
        label='Upload Seismic Data',
    )

    model = forms.ChoiceField(
        choices = MODEL_CHOICES,
        required = True,
        label = "Select Type",
        widget = forms.Select(attrs = {
            "class": "form-control"
        })
    )