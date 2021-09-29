import yaml

with open('test.yaml') as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)

print(type(conf))
print(conf)

def print_dict(idx_list, **args):
    print(idx_list)
    for key, value in args.items():
        print(key, value)

idx_list = [0, 1, 2]
print_dict(idx_list, **conf)