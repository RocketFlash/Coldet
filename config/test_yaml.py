import yaml

with open("unet.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

print(cfg['input_shape'])
print(cfg['loss'])

print(type(cfg['metrics']))
print(type(cfg['loss']))

print(type(cfg))