import pandas as pd
import datetime
import torch
import sys
import os
from setfit import SetFitModel, SetFitTrainer
from torchviz import make_dot


current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from utils.sentence_classification import evaluate, save_model, create_setfit_logger, load_datasets
from scripts.sentence_classification.data_handler import data_handler

dfs = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CosineSimilarityLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, x1, x2, target):
        cos_sim = torch.nn.functional.cosine_similarity(x1, x2)
        loss = 1 - cos_sim
        if target == 0:
            loss = cos_sim
        return loss.mean()
    
class SetfitTrainerWrapper:
    def __init__(self, logger, train_path: str = None,
                 save_dir: str = None,
                 num_samples_list: int = None,
                 model_name_initial: str = None,
                 all_class: bool = None,
                 batch_size: int = None,
                 num_iteration: int = None,
                 labels_: list = None,
                 positive_number: int = 8,
                 pretrained_model: str = None,
                 result_path: str = None,
                 load_xlsx: bool = None,
                 SEED: int = 7,
                 pretrained_model_list: str = None,
                 val_path:str = None,
                 generated_data:bool = None):
        
        self.train_path = train_path
        self.val_path = val_path
        self.save_dir = save_dir
        self.result_path = result_path
        self.num_samples_list = num_samples_list
        self.model_name_initial = model_name_initial
        self.all_class = all_class
        self.batch_size = batch_size
        self.num_iteration = num_iteration
        self.labels_ = labels_
        self.pretrained_model = pretrained_model
        self.positive_number = positive_number
        self.handler = data_handler(data_path=train_path,
                                    positive_number=positive_number,
                                    labels_=labels_,
                                    load_xlsx=load_xlsx,
                                    SEED=SEED)
        self.logger = logger
        self.pretrained_model_list = pretrained_model_list
        self.generated_data = generated_data
    

    def train_by_case(self, label: str = None, verdict_test: list = None):

        
        if not self.generated_data:
            handler = data_handler(data_path=self.train_path,
                                   positive_number=50,
                                   labels_=self.labels_,
                                   load_xlsx=True,
                                   SEED=7)
            df_test = handler.dfs[label]
            df_test['case'] = handler.dfs['case']

            val_df = df_test[df_test['case'].isin(verdict_test)]
            val_df['text'] = val_df['text'].astype(str).dropna()
            val_df = val_df.dropna()

            df = self.handler.dfs[label]
            df['text'] = df['text'].astype(str)
            df['case'] = self.handler.dfs['case']
            df = df.dropna()

            train_df = df[~df['case'].isin(verdict_test)]

            self.positive_number = train_df[train_df['label'] == 1].sum()['label']

            train_df = train_df.drop('case', axis=1)
            val_df = val_df.drop('case', axis=1)
            
        else:
            train_df.columns = ['text', 'label']
            val_df.columns = ['text', 'label']
            
        train_df = train_df.groupby('label').sample(n=200, random_state=42)

        train_ds, val_ds = load_datasets(train_df,
                                         val_df)

        model = SetFitModel.from_pretrained(self.pretrained_model).to(device)
        model_name = str(label) + '-' + str(self.model_name_initial)

        logger_w = create_setfit_logger(model_name,
                                        1,
                                        self.positive_number,
                                        self.batch_size,
                                        self.num_iteration)

        trainer = SetFitTrainer(
            model=model.to(device),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            batch_size=self.batch_size,
            num_iterations=self.num_iteration,
            num_epochs=1,
            loss_class=logger_w
        )


        trainer.train()
        torch.cuda.empty_cache()

        precision, recall, f1, auc_pr, model, val_df = evaluate(model, val_df, str(verdict_test) + '.csv', label)
        metrics_path = os.path.join(self.save_dir, f"{label}_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")
            f.write(f"AUC-PR: {auc_pr}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Validation pos num DataFrame: {val_df['label'].value_counts().values[1]}\n")

        return precision, recall, f1, auc_pr, model, val_df


    def train(self, experiment_name, load_xlsx):
        today = datetime.date.today().strftime('%Y-%m-%d-%H-%M')
        save_dir_name = today + '_' + experiment_name
        save_path = os.path.join(self.save_dir, save_dir_name)
        for pretrained_model in self.pretrained_model_list:

            results = {
                "label": [],
                "name": [],
                "precision": [],
                "recall": [],
                "F1-score": [],
                "AUC-PR": [],
                'Num_test_sentences': [],
                'Num_positive_labels': []
            }
            if load_xlsx:
                df = pd.read_excel(self.train_path)
            else:
                df = pd.read_csv(self.train_path)
            
            df.rename(columns={
                            'verdict': 'Case',
                            'text': 'Text'
                            }, inplace=True)
            tagged_cases = sorted(list(set(df['Case'].values)))
            verdict_sets=[]

            for label in self.labels_:
                self.logger.info(f"ֿ\nStart to train {label} classifier\n")
                start_idx = 0
                while start_idx < len(tagged_cases):
                    verdict_test = tagged_cases[start_idx:start_idx + 4]
                    verdict_sets.append(verdict_test)

                    precision, recall, f1, auc_pr, model, val_df = self.train_by_case(label=label,
                                                                                      verdict_test=verdict_test)
                    dfs.append((val_df, str(verdict_test)))
                    results['label'].append(label)
                    results['name'].append(str(verdict_test))
                    results['precision'].append(precision)
                    results['recall'].append(recall)
                    results['F1-score'].append(f1)
                    results['AUC-PR'].append(auc_pr)
                    results['Num_test_sentences'].append(len(val_df))
                    try:
                        results['Num_positive_labels'].append(val_df['label'].value_counts().values[1])
                    except:
                        print()
                    
                    save_model(save_path, label, str(verdict_test), model, "set-fit", experiment_name)
                    start_idx += 7
                    
                    break
                
            df = pd.DataFrame(results)
            file_name = f"setfit_full_report.csv"
            result_path = os.path.join(save_path ,file_name)
            df.to_csv(result_path)

            agg_df = df.groupby('label').mean().reset_index()
            file_name = f"setfit_agg_report.csv"
            result_path = os.path.join(save_path ,file_name)

            agg_df.to_csv(result_path)
            
            self.logger.info(f'The models and their evaluation were successfully saved in {save_path}')

        torch.cuda.empty_cache()
        return save_path

        # precision, recall, f1, auc_pr, model, val_df = evaluate(model, val_df, str(verdict_test) + '.csv', label)
        # return precision, recall, f1, auc_pr, model, val_df


