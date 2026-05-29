# ********** Mon programme de formation du générateur Electra 
# Last update :

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from matplotlib import testing
from typing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
from torch import nn
from torch import cuda 

from pytorch_lightning.tuner import Tuner
# **********
# Last update : 

# My data path
dataset_path = "/home/habiba_c/MesProgsPython/data/full_dataset/"

file_paths = list(Path(dataset_path).glob("*.csv"))

dfs = []
for file_path in file_paths:
  dfs.append(pd.read_csv(file_path))
df = pd.concat(dfs)
df.created_utc = pd.to_datetime(df.created_utc, unit='s') # This is not mandatory
df.head()

# **********
# Since multiple annotators do not necessarily agree on the emotion label,
# we're going to take the most frequent emotion as the real label:

texts = []
emotions = []

emotion_categories = df.columns[9:]
#emotion_categories

for comment_id, group in tqdm(df.groupby("id")):
  texts.append(group.iloc[0].text)
  emotions.append(group[emotion_categories].sum(axis=0).argmax())

# **********
text_df = pd.DataFrame({"text": texts, "emotion": emotions})
#text_df.head()

print("I am here! FIN DE TRAITEMENT DU DATASET")

# **********

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import ElectraTokenizerFast as ElectraTokenizer,ElectraPreTrainedModel,ElectraModel,AdamW
from transformers.activations import get_activation


class EmotionDataModule(pl.LightningDataModule):
  def __init__(
      self, data: pd.DataFrame, tokenizer: ElectraTokenizer, batch_size: int
  ):
    super().__init__()
    self.data = data
    self.tokenizer = tokenizer
    self.batch_size = batch_size

  def setup(self, stage: Optional[str] = None):
    self.train_df, test_df = train_test_split(self.data, test_size=0.2)
    self.val_df, self.test_df = train_test_split(test_df, test_size=0.5)

  def train_dataloader(self):
    return DataLoader(
        dataset=EmotionDataset(self.train_df, self.tokenizer),
        batch_size=self.batch_size,
        num_workers=os.cpu_count(),
        shuffle=True
    )

  def val_dataloader(self):
    return DataLoader(
        dataset=EmotionDataset(self.val_df, self.tokenizer),
        batch_size=self.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False
    )

  def test_dataloader(self):
    return DataLoader(
        dataset=EmotionDataset(self.test_df, self.tokenizer),
        batch_size=self.batch_size,
        num_workers=os.cpu_count(),
        shuffle=False
    )

# **********

print("I am here! EmotionDataModule")

MODEL_NAME = "google/electra-base-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

data_module = EmotionDataModule(text_df, tokenizer, batch_size=32)
data_module.setup()

# **********

class EmotionDataset(Dataset):
  def __init__(self, data: pd.DataFrame, tokenizer: ElectraTokenizer):
    self.data = data
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    row = self.data.iloc[idx]
    encoding = tokenizer(
        row.text,
        max_length=64,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "label": torch.tensor(row.emotion)
    }

ds = EmotionDataset(text_df, tokenizer)
#print(len(ds) == len(text_df))

#for item in ds:
  #print(item["input_ids"][:10])
  #print(item["label"])
  #break


# **********

#for batch in data_module.train_dataloader():
  #print(len(batch))
  #print(batch["input_ids"].shape, batch["attention_mask"].shape, batch["label"].shape)
  #break

print("I am here! inputIds,attentionMask,label")

# Emotion Classifier

class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

print("I am here! EmotionClassifierHead")

# **********
# from transformers.utils.dummy_pt_objects import ElectraPreTrainedModel

class ElectraClassifier(ElectraPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.n_classes = config.num_labels
    self.config = config
    self.electra = ElectraModel(config)
    self.classifier = ElectraClassificationHead(config)

    self.post_init()

  def forward(
      self,
      input_ids=None,
      attention_mask=None
  ):
    discriminator_hidden_states = self.electra(input_ids, attention_mask)
    sequence_output = discriminator_hidden_states[0]
    logits = self.classifier(sequence_output)
    return logits

print("I am here! ElectraClassifier")

# **********
class EmotionClassifier(pl.LightningModule):
  def __init__(self, n_classes, learning_rate: Optional[float]=None):
    super().__init__()
    self.n_classes = n_classes
    self.classifier = ElectraClassifier.from_pretrained(
        "google/electra-base-discriminator",
        #"google/electra-base-discriminator",
        num_labels=n_classes
    )
    self.criterion = nn.CrossEntropyLoss()
    self.learning_rate = learning_rate

  def forward(self, input_ids, attention_mask):
    return self.classifier(input_ids, attention_mask)

  def run_step(self, batch, stage):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"].long()
    logits = self(input_ids, attention_mask)

    loss = self.criterion(logits, labels)
    self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def training_step(self, batch, batch_idx):
    return self.run_step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self.run_step(batch, "val")

  def test_step(self, batch, batch_idx):
    return self.run_step(batch, "test")

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=self.learning_rate)

# Commented out IPython magic to ensure Python compatibility.

print("I am here! EmotionClassifier")

# **********

MAX_LEARNING_RATE = 1e-2
# BATCH_SIZE = 512 
# TRAINING_STEPS = 650

BATCH_SIZE = 128 
TRAINING_STEPS = -1

data_module = EmotionDataModule(
    text_df,
    tokenizer,
    batch_size=BATCH_SIZE
)

model = EmotionClassifier(
    n_classes = len(emotion_categories),
    learning_rate = 0.0001
)

print("I am here! model, data module")

# **********
# %env CUDA_VISIBLE_DEVICES=5 avec notebook
# $ export CUDA_VISIBLE_DEVICES=0,1 en ligne de commande

print("I am here! cuda, device")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# trainer = pl.Trainer(devices=0, accelerator="gpu")
# trainer = pl.Trainer(devices=3, accelerator="auto")

# Commented out IPython magic to ensure Python compatibility.
# **********
# **********
# %load_ext tensorboard
# load_ext tensorboard
# **********

# logger = TensorBoardLogger(save_dir=experiments_dir, name="emotion_classifica$

# **********
# %tensorboard --logdir experiments_dir

print("I am here!max learning, batch size, training steps")

# *********

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
experiments_dir = "/home/habiba_c/MesProjets2024/experiments/"

model_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{val_loss:.2f}",
    save_last=True,
    save_top_k=3,
    monitor="val_loss_epoch",
    mode="min"
)

# **********
trainer = pl.Trainer(
    default_root_dir=experiments_dir,
    accelerator="gpu",devices=1,
   #  max_epochs=16,
    max_epochs=32,
    max_steps=TRAINING_STEPS,
    precision=16,
    val_check_interval=40,
    callbacks=[
        model_checkpoint
    ],
    #logger=logger
)

# **********Rechercher le taux d'apprentissage optimal 

#tuner = Tuner(trainer)
#lr_finder = tuner.lr_find(model, data_module, max_lr=MAX_LEARNING_RATE)

# **********
#print("I am here! les résultats de recherche du lrFinder")

#lr_finder.results

# **********
# !nvidia-smi

# **********
#fig = lr_finder.plot(suggest=True)

# **********

#new_lr = lr_finder.suggestion()

#print("I am here! le nouveau lr = ",new_lr)


# Commented out IPython magic to ensure Python compatibility.
# **********
# model.learning_rate = new_lr
# model.learning_rate = 0.0001

#model.learning_rate = 0.00013803842646028838
model.learning_rate = 0.0003630780547701014

# **********
print("I am here! AVANT LE TRAIN lr = ",model.learning_rate)
 
trainer.fit(model, data_module)

print("I am here! APRES LE TRAIN lr = ",model.learning_rate)

# **********
# trainer.test(datamodule=data_module,ckpt_path='best')

print("I am here! APRES LE TEST")

# **********
# model.classifier.save_pretrained("emotion_classifier")

# **********
# best_model_path = trainer.checkpoint_callback.best_model_path

# **********
# trainer.checkpoint_callback.best_model_path

# **********
#best_model_path = "/home/habiba_c/MesProgsPython/" +  best_model_path

#best_model_path

print("I am here! LA FIN")





