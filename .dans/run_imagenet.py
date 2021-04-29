"""Run ImageNet
"""

import sys
sys.path.append('/home/dans/flax/examples/imagenet')

import train
from configs import default as config_lib

config = config_lib.get_config()
config.dataset = 'imagenette'
config.model = 'ResNet18'
config.half_precision = True
batch_size = 512
config.learning_rate *= batch_size / config.batch_size
config.batch_size = batch_size
print(config)


# Train from scratch
# Takes ~1.5 min / epoch.
for num_epochs in (5, 10):
  config.num_epochs = num_epochs
  config.warmup_epochs = config.num_epochs / 10
  name = f'{config.model}_{config.learning_rate}_{config.num_epochs}'
  print(f'\n\n{name}')
  state = train.train_and_evaluate(config, workdir=f'./models/{name}')
