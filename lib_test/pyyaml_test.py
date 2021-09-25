import yaml

with open('test.yaml') as f:
    conf = yaml.load(f)

print(type(conf))
print(conf)