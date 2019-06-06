import yaml
import losses
from keras.optimizers import Adam

def parse_train_params(filename='unet.yml'):
    params = {}
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    metrics = []
    loss = []
    for m in cfg['metrics']:
        if m == 'accuracy':
            metrics.append('accuracy')
        elif m == 'dice':
            metrics.append(losses.dice)
    for l in cfg['loss']:
        if l == 'binary_crossentropy': 
            loss.append('binary_crossentropy')
        elif l == 'dice_and_iou':
            loss.append(losses.dice_and_iou)
        elif l == 'dice_and_bce':
            loss.append(losses.dice_and_binary_crossentropy)
    if cfg['optimizer'] == 'adam':
        optimizer = Adam()
    params = {k:v for k,v in cfg.items() if k not in ['metrics', 'loss', 'optimizer']}
    params['metrics'] = metrics
    params['loss'] = loss
    params['optimizer'] = optimizer
    return params