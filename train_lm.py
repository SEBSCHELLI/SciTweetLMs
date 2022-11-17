import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from src.data import LMDataLoader


# define parameters and set up wandb
run = wandb.init(project="SciTweetsLM_Training")

model_config = {'pretrained_model_name_or_path': 'vinai/bertweet-base'}
tokenizer_config = {'pretrained_model_name_or_path': 'vinai/bertweet-base', 'max_len': 128}

#model_config = {'pretrained_model_name_or_path': 'allenai/scibert_scivocab_uncased'}
#tokenizer_config = {'pretrained_model_name_or_path': 'allenai/scibert_scivocab_uncased', 'max_len': 128}

wandb.config.update(tokenizer_config)

preprocessing_config = {'lowercase': True,
                        'normalize': True,
                        'urls': False,
                        'user_handles': '@USER',
                        'emojis': 'demojize'}

wandb.config.update(preprocessing_config)

run_name = "bertweet_test"
wandb.config['run_name'] = run_name

path = "training_data/tweets_en_sci_urls.pkl"
wandb.config['path'] = path

split = 0.8
wandb.config['split'] = split

seed = 0
wandb.config['seed'] = seed

epochs = 10
wandb.config['epochs'] = epochs

batch_size = 24
wandb.config['batch_size'] = batch_size


print('load model and tokenizer')
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
model = AutoModelForMaskedLM.from_pretrained(**model_config).cuda()

print('load data')
train_data, test_data = LMDataLoader.get_train_and_test_data(path, tokenizer, preprocessing_config, split, seed)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=f"./output/{run_name}",
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_strategy='steps',
    save_steps=5000,
    seed=seed,
    no_cuda=False,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=test_data
)

print('start training')
trainer.train()
