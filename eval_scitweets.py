import os
import copy
import numpy as np
import pandas as pd
import wandb
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, average_precision_score, roc_auc_score

from src.data import SciTweetsDataLoader

def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x), axis=1, keepdims=True)
    return exp / exp_sum

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def annotate_test_dataframe(pred_output):
    if num_labels == 2:

        test_df[f'{cat}_logits'] = pred_output.predictions[:, 0]
        test_df[f'{cat}_pred'] = np.argmax(pred_output.predictions, 1)
        test_df[f'{cat}_score'] = softmax(pred_output.predictions)[:, 0]

    elif num_labels == 3:

        test_df['cat1_logits'] = pred_output.predictions[:, 0]
        test_df['cat2_logits'] = pred_output.predictions[:, 1]
        test_df['cat3_logits'] = pred_output.predictions[:, 2]

        predictions = (pred_output.predictions > 0) * 1
        test_df['cat1_pred'] = predictions[:, 0]
        test_df['cat2_pred'] = predictions[:, 1]
        test_df['cat3_pred'] = predictions[:, 2]

        test_df['cat1_score'] = sigmoid(pred_output.predictions[:, 0])
        test_df['cat2_score'] = sigmoid(pred_output.predictions[:, 1])
        test_df['cat3_score'] = sigmoid(pred_output.predictions[:, 2])

    test_df['fold'] = fold

    annotated_test_data.append(test_df.copy())


def compute_fold_metrics(pred_output):
    epoch = trainer.state.epoch
    fold_epoch = int(epoch + (fold * epochs))
    metrics = {'fold_epoch': fold_epoch}

    labels = pred_output.label_ids

    if num_labels == 2:
        predictions = np.argmax(pred_output.predictions, 1)
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        rec = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        metrics.update({
            f'{cat}_acc': acc, f'{cat}_prec': prec, f'{cat}_rec': rec, f'{cat}_f1': f1})

    elif num_labels == 3:
        predictions = (pred_output.predictions > 0) * 1

        cat1_acc = accuracy_score(labels[:, 0], predictions[:, 0])
        cat2_acc = accuracy_score(labels[:, 1], predictions[:, 1])
        cat3_acc = accuracy_score(labels[:, 2], predictions[:, 2])

        cat1_pre = precision_score(labels[:, 0], predictions[:, 0])
        cat2_pre = precision_score(labels[:, 1], predictions[:, 1])
        cat3_pre = precision_score(labels[:, 2], predictions[:, 2])

        cat1_re = recall_score(labels[:, 0], predictions[:, 0])
        cat2_re = recall_score(labels[:, 1], predictions[:, 1])
        cat3_re = recall_score(labels[:, 2], predictions[:, 2])

        cat1_f1 = f1_score(labels[:, 0], predictions[:, 0])
        cat2_f1 = f1_score(labels[:, 1], predictions[:, 1])
        cat3_f1 = f1_score(labels[:, 2], predictions[:, 2])

        metrics.update({
            f'cat1_acc': cat1_acc, f'cat2_acc': cat2_acc, f'cat3_acc': cat3_acc,
            f'cat1_pre': cat1_pre, f'cat2_pre': cat2_pre, f'cat3_pre': cat3_pre,
            f'cat1_re': cat1_re, f'cat2_re': cat2_re, f'cat3_re': cat3_re,
            f'cat1_f1': cat1_f1, f'cat2_f1': cat2_f1, f'cat3_f1': cat3_f1})

    if epoch == epochs:
        annotate_test_dataframe(pred_output)

    return metrics


def compute_overall_metrics(data):
    overall_metrics = {}

    def get_pr_table(labels, scores):
        precs, recs, thresholds = precision_recall_curve(labels, scores)
        pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precs[:-1], 'recall': recs[:-1]})
        pr_df = pr_df.sample(n=min(1000, len(pr_df)), random_state=0)
        pr_df = pr_df.sort_values(by='threshold')
        pr_table = wandb.Table(dataframe=pr_df)
        return pr_table

    if num_labels == 2:
        scores = data[f'{cat}_score']
        preds = data[f'{cat}_pred'] == 1
        labels = data[f'labels'] == 1

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        aps = average_precision_score(labels, scores)
        roc_auc = roc_auc_score(labels, scores)
        overall_metrics.update({f'{cat}_avg_acc': acc,
                                f'{cat}_avg_prec': prec,
                                f'{cat}_avg_rec': rec,
                                f'{cat}_avg_f1': f1,
                                f'{cat}_avg_aps': aps,
                                f'{cat}_avg_roc_auc': roc_auc,
                                })

        # log pr-curve
        pr_table = get_pr_table(labels, scores)
        overall_metrics.update({f'{cat}_pr_table': pr_table})

    elif num_labels == 3:

        for _cat in ['cat1', 'cat2', 'cat3']:
            scores = data[f'{_cat}_score']
            preds = data[f'{_cat}_pred'] == 1
            labels = data[f'{_cat}_final_answer'] == 1

            acc = accuracy_score(labels, preds)
            prec = precision_score(labels, preds)
            rec = recall_score(labels, preds)
            f1 = f1_score(labels, preds)
            aps = average_precision_score(labels, scores)
            roc_auc = roc_auc_score(labels, scores)
            overall_metrics.update({f'{_cat}_avg_acc': acc,
                                    f'{_cat}_avg_prec': prec,
                                    f'{_cat}_avg_rec': rec,
                                    f'{_cat}_avg_f1': f1,
                                    f'{_cat}_avg_aps': aps,
                                    f'{_cat}_avg_roc_auc': roc_auc,
                                    })

            # log pr-curve
            pr_table = get_pr_table(labels, scores)
            overall_metrics.update({f'{_cat}_pr_table': pr_table})

    return overall_metrics


if __name__ == '__main__':

    run = wandb.init(project="SciTweetsLM_Eval_SciTweets")

    model_id_or_path = 'vinai/bertweet-base'
    #model_id_or_path = 'allenai/scibert_scivocab_uncased'
    #model_id_or_path = 'cardiffnlp/twitter-roberta-base-sep2022'
    model_id_or_path = 'digitalepidemiologylab/covid-twitter-bert-v2'
    model_id_or_path = '/data_ssds/disk11/slinzbach/socioroberta'
    wandb.config['model_id_or_path'] = model_id_or_path


    tokenizer_id_or_path = 'vinai/bertweet-base'
    #tokenizer_id_or_path = 'allenai/scibert_scivocab_uncased'
    #tokenizer_id_or_path = 'cardiffnlp/twitter-roberta-base-sep2022'
    tokenizer_id_or_path = 'digitalepidemiologylab/covid-twitter-bert-v2'
    tokenizer_id_or_path = '/data_ssds/disk11/slinzbach/socioroberta'
    wandb.config['tokenizer_id_or_path'] = tokenizer_id_or_path

    tokenizer_max_len = 128
    wandb.config['tokenizer_max_len'] = tokenizer_max_len

    n_folds = 10
    wandb.config['n_folds'] = n_folds

    epochs = 10
    wandb.config['epochs'] = epochs

    seed = 0
    wandb.config['seed'] = seed

    dataloader_config = {'per_device_train_batch_size': 32,
                         'per_device_eval_batch_size': 64}
    wandb.config.update(dataloader_config)

    preprocessing_config = {'lowercase': True,
                            'normalize': True,
                            'urls': 'original_urls',
                            'user_handles': '@USER',
                            'emojis': 'demojize'}
    wandb.config.update(preprocessing_config)

    learning_rate = 5e-5
    wandb.config['learning_rate'] = learning_rate

    cat = "cat1"  # or multilabel or scirelated
    wandb.config['cat'] = cat

    if cat in ['cat1', 'cat2', 'cat3', 'scirelated']:
        num_labels = 2
        problem_type = "single_label_classification"

    elif cat == "multilabel":
        num_labels = 3
        problem_type = "multi_label_classification"

    wandb.config['num_labels'] = num_labels
    wandb.config['problem_type'] = problem_type

    num_layers_freezed = False
    wandb.config['num_layers_freezed'] = num_layers_freezed

    print('load model and tokenizer')
    tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                        'max_len': tokenizer_max_len}
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

    model_config = {'pretrained_model_name_or_path': model_id_or_path,
                    'num_labels': num_labels,
                    'problem_type': problem_type}
    model = AutoModelForSequenceClassification.from_pretrained(**model_config)

    if num_layers_freezed:
        if "BertForSequenceClassification" in type(model):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

            for n, param in model.bert.encoder.named_parameters():
                for i in range(num_layers_freezed):
                    if f'layer.{i}.' in n:
                        param.requires_grad = False

        elif "RobertaForSequenceClassification" in type(model):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False

            for n, param in model.roberta.encoder.named_parameters():
                for i in range(num_layers_freezed):
                    if f'layer.{i}.' in n:
                        param.requires_grad = False

    print(f'load data')
    dl = SciTweetsDataLoader(cat, tokenizer, n_folds, preprocessing_config, seed)

    annotated_test_data = []

    for fold in range(n_folds):
        print(f'Fold {fold+1}')
        fold_model = copy.deepcopy(model).cuda()
        #fold_model = copy.deepcopy(model)

        train_dataset, test_dataset = dl.get_datasets_for_fold(fold)
        test_df = dl.get_test_df_for_fold(fold).copy()

        training_args = TrainingArguments(
            output_dir="results",  # output directory
            optim='adamw_torch', #### MAYBE performance better if removed
            num_train_epochs=epochs,  # total number of training epochs
            **dataloader_config,
            warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            learning_rate=learning_rate,
            logging_dir='./logs',  # directory for storing logs
            logging_strategy='epoch',
            save_strategy='no',
            evaluation_strategy="epoch",  # evaluate each `logging_steps`
            no_cuda=False,
            report_to='wandb'
        )

        trainer = Trainer(
            model=fold_model,  # the instantiated Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=test_dataset,
            compute_metrics=compute_fold_metrics
        )

        trainer.train()
        print('***** Finished Training *****\n\n\n')

    print('Evaluate all folds')
    data = pd.concat(annotated_test_data)
    data.to_csv(f'output/scitweets/{run.name}_preds.tsv', index=False, sep='\t')
    metrics = compute_overall_metrics(data)
    wandb.log(metrics)
