from django import template
from django.template.defaultfilters import stringfilter
import markdown as md

register = template.Library()


@register.filter()
@stringfilter
def markdown(value):
    """
    Converts Markdown to HTML
    """
    return md.markdown(value, extensions=[
        'markdown.extensions.fenced_code',
        'markdown.extensions.codehilite',
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists',
    ])
