cd /home/dans/flax/examples/imagenet
python


example_directory = 'examples/imagenet'
editor_relpaths = ('configs/default.py', 'input_pipeline.py', 'models.py', 'train.py')

repo, branch = 'https://github.com/google/flax', 'master'



import json
from absl import logging
import flax
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

logging.set_verbosity(logging.INFO)





# Helper functions for images.

def show_img(img, ax=None, title=None):
  """Shows a single image."""
  if ax is None:
    ax = plt.gca()
  img *= tf.constant(input_pipeline.STDDEV_RGB, shape=[1, 1, 3], dtype=img.dtype)
  img += tf.constant(input_pipeline.MEAN_RGB, shape=[1, 1, 3], dtype=img.dtype)
  img = np.clip(img.numpy().astype(int), 0, 255)
  ax.imshow(img)
  ax.set_xticks([])
  ax.set_yticks([])
  if title:
    ax.set_title(title)

def show_img_grid(imgs, titles):
  """Shows a grid of images."""
  n = int(np.ceil(len(imgs)**.5))
  _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
  for i, (img, title) in enumerate(zip(imgs, titles)):
    show_img(img, axs[i // n][i % n], title)




import input_pipeline
import models
import train
from configs import default as config_lib



dataset_builder = tfds.builder('imagenette')
dataset_builder.download_and_prepare()
ds = dataset_builder.as_dataset('train')
dataset_builder.info




with open('mapping_imagenet.json') as f:
    mapping_imagenet = json.load(f)

# Mapping imagenette label name to imagenet label index.
imagenette_labels = {
d['v3p0']: d['label']
for d in mapping_imagenet
}

# Mapping imagenette label name to human-readable label.
imagenette_idx = {
    d['v3p0']: idx
    for idx, d in enumerate(mapping_imagenet)
}

def imagenette_label(idx):
  """Returns a short human-readable string for provided imagenette index."""
  net = dataset_builder.info.features['label'].int2str(idx)
  return imagenette_labels[net].split(',')[0]

def imagenette_imagenet2012(idx):
  """Returns the imagenet2012 index for provided imagenette index."""
  net = dataset_builder.info.features['label'].int2str(idx)
  return imagenette_idx[net]

def imagenet2012_label(idx):
  """Returns a short human-readable string for provided imagenet2012 index."""
  return mapping_imagenet[idx]['label'].split(',')[0]




train_ds = input_pipeline.create_split(
    dataset_builder, 128, train=True,
)
eval_ds = input_pipeline.create_split(
    dataset_builder, 128, train=False,
)



config = config_lib.get_config()
config.dataset = 'imagenette'
config.model = 'ResNet18'
config.half_precision = True
batch_size = 512
config.learning_rate *= batch_size / config.batch_size
config.batch_size = batch_size
config



# Regenerate datasets with updated batch_size.
train_ds = input_pipeline.create_split(
    dataset_builder, config.batch_size, train=True,
)
eval_ds = input_pipeline.create_split(
    dataset_builder, config.batch_size, train=False,
)


################## Train from scratch 

# Takes ~1.5 min / epoch.
for num_epochs in (5, 10):
  config.num_epochs = num_epochs
  config.warmup_epochs = config.num_epochs / 10
  name = f'{config.model}_{config.learning_rate}_{config.num_epochs}'
  print(f'\n\n{name}')
  state = train.train_and_evaluate(config, workdir=f'./models/{name}')


####################################### Load pretrained model

# Load model checkpoint from cloud.
! wget https://storage.googleapis.com/flax_public/examples/imagenet/v100_x8/checkpoint_250200

#from flax.training import checkpoints
#import tensorflow as tf
#import os
config_name = 'v100_x8'
#pretrained_path = f'gs://flax_public/examples/imagenet/{config_name}'
#latest_checkpoint = checkpoints.natural_sort(
#    tf.io.gfile.glob(f'{pretrained_path}/checkpoint_*'))[0]
#if not os.path.exists(os.path.basename(latest_checkpoint)):
#  tf.io.gfile.copy(latest_checkpoint, os.path.basename(latest_checkpoint))
#

# Load config that was used to train checkpoint.
import importlib
config = importlib.import_module(f'configs.{config_name}').get_config()


# Load models & state (takes ~1 min to load the model).
model_cls = getattr(models, config.model)
model = train.create_model(
    model_cls=model_cls, half_precision=config.half_precision)
state = train.create_train_state(
    jax.random.PRNGKey(0), config, model, image_size=input_pipeline.IMAGE_SIZE)
state = train.restore_checkpoint(state, './')

############################ Inference


# Load batch from imagenette eval set.
batch = next(iter(eval_ds))
{k: v.shape for k, v in batch.items()}


# Evaluate using model trained on imagenet.
logits = model.apply({'params': state.optimizer.target, **state.model_state}, batch['image'][:128], train=False)

# Find classification mistakes.
preds_labels = list(zip(logits.argmax(axis=-1), map(imagenette_imagenet2012, batch['label'])))
error_idxs = [idx for idx, (pred, label) in enumerate(preds_labels) if pred != label]
error_idxs



# The mistakes look all quite reasonable.
show_img_grid(
    [batch['image'][idx] for idx in error_idxs[:9]],
    [f'pred: {imagenet2012_label(preds_labels[idx][0])}\n'
     f'label: {imagenet2012_label(preds_labels[idx][1])}'
    for idx in error_idxs[:9]],
)
plt.tight_layout()


# Define parallelized inference function in separate cell so the the cached
# compilation can be used if below cell is executed multiple times.
@jax.pmap
def p_get_logits(images):
  return model.apply({'params': state.optimizer.target, **state.model_state},
                     images, train=False)

eval_iter = train.create_input_iter(dataset_builder, config.batch_size,
                                    input_pipeline.IMAGE_SIZE, tf.float32,
                                    train=False, cache=True)


# Compute accuracy.
eval_steps = dataset_builder.info.splits['validation'].num_examples // config.batch_size
count = correct = 0
for step, batch in zip(range(eval_steps), eval_iter):
  labels = [imagenette_imagenet2012(label) for label in batch['label'].flatten()]
  logits = p_get_logits(batch['image'])
  logits = logits.reshape([-1, logits.shape[-1]])
  print(f'Step {step+1}/{eval_steps}...')
  count += len(labels)
  correct += (logits.argmax(axis=-1) == jnp.array(labels)).sum()

correct / count