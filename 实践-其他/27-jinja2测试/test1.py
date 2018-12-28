from jinja2 import Template
template = Template('Hello {{ name }}!')
a = template.render(name='John Doe')
print(a)