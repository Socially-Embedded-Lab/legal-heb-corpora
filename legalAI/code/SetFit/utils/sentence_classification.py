import os
import datetime
import sys
import pandas as pd
import wandb
from torch import nn
from sentence_transformers.losses import CosineSimilarityLoss
from dataclasses import dataclass
from typing import Callable
import functools
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, auc, precision_recall_curve
from typing import Callable

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, pred_sentencing_path)

from scripts.sentence_classification.predict_sentence_cls.loads import load_all_classifies
from utils.errors.predict_sentence_cls import classifiers_path_error


def save_model(save_dir, label, model_name, model, train_model_name, experiment_name):
    """

    Save the trained SetFit model and related results to a specified directory. 
    The model is saved using the '_save_pretrained' method of the trainer's model,
    in a 'model' subdirectory within a date-named folder. If a folder with the same date and model name exists, 
    a version number is appended to the folder name.


    Parameters:
        trainer: The SetFitTrainer object containing the trained model.
        model_name (str): The name of the model to be saved.

    """
    
    # Create date folder
    today = datetime.date.today().strftime('%Y-%m-%d')
    version = 1
    save_path = f"{save_dir}/{train_model_name}/{label}/{model_name}"
    while os.path.exists(save_path):
        version += 1
        save_path = f"{save_dir}/{train_model_name}/{label}/{today}-v{version}/{model_name}"
        
        # Save to CSV
        data = {
            'Experiment Name': [experiment_name],
            'Label': [label],
            'Model Name': [train_model_name],
            'Version': [version]
        }

        df = pd.DataFrame(data)
        csv_path = os.path.join(save_dir, 'experiment_names.csv')

        # Check if the CSV exists
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)

            # Filter rows that match the triple
            mask = (df_existing['Experiment Name'] == experiment_name) & (df_existing['Label'] == label) & (df_existing['Model Name'] == train_model_name) & (df_existing['Version'] == version)

        # If the row does not exist in the CSV
            if not mask.any():
                df = pd.concat([df_existing, df], ignore_index=True)
                df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)
    else:
        os.makedirs(save_path)

    # Save model
    model.save_pretrained(save_path)



def create_setfit_logger(model_name, num_epochs, num_samples, batch_size, num_iterations):
    """Create a Hugging Face Trainer logger that logs training loss and metrics to Weights & Biases."""

    @dataclass
    class LoggingWrapper:
        loss_class: Callable
        num_epochs: int
        num_samples: int
        batch_size: int
        num_iterations: int
        
        def __post_init__(self):
            # Add the __name__ attribute dynamically
            self.__name__ = 'LoggingWrapper'

        def __call__(self, *args, **kwargs):
            wandb.init(project="setfit", name=model_name)
            wandb.config.update({"num_epochs": self.num_epochs, "num_samples": self.num_samples,
                                 "batch_size": self.batch_size, "num_iterations": self.num_iterations},
                                allow_val_change=True)
            loss_class_instance = self.loss_class(*args, **kwargs)
            loss_class_instance.forward = self.log_forward(loss_class_instance.forward)
            # loss_class_instance.validate = self.log_validation_loss(loss_class_instance.validate)
            return loss_class_instance

        def log_forward(self, forward_func: Callable):
            @functools.wraps(forward_func)
            def log_wrapper_forward(*args, **kwargs):
                loss = forward_func(*args, **kwargs)
                wandb.log({"training_loss": loss.item(), "num_epochs": self.num_epochs, "num_samples": self.num_samples,
                           "batch_size": self.batch_size, "num_iterations": self.num_iterations})
                return loss

            return log_wrapper_forward

        def log_validation_loss(self, validate_func: Callable):
            @functools.wraps(validate_func)
            def log_wrapper_validate(*args, **kwargs):
                loss = validate_func(*args, **kwargs)
                wandb.log(
                    {
                        "validation_loss": loss,
                        "num_epochs": self.num_epochs,
                        "num_samples": self.num_samples,
                        "batch_size": self.batch_size,
                        "num_iterations": self.num_iterations,
                    }
                )
                return loss

            return log_wrapper_validate

    return LoggingWrapper(CosineSimilarityLoss, num_epochs, num_samples, batch_size, num_iterations)


def load_datasets(train_df, val_df):
    """Load the training and validation datasets from the Pandas DataFrames."""
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return train_ds, val_ds

def evaluate(tagged_df, df_predictions, df_probabilities, column, file, label=None, logger=None):

        """
        Evaluate the SetFit model on the validation dataset and save the results.

        This method evaluates the SetFit model on the provided validation dataset and computes various evaluation metrics.
        The evaluation results are saved in a CSV file with the specified 'name' in the given directory.
        The evaluation metrics include precision, recall, F1-score, and AUC-PR (Area Under the Precision-Recall curve).

        Parameters:
            model: The trained SetFit model to be evaluated.
            tagged_df (pd.DataFrame): The validation DataFrame containing the validation dataset.
            name (str): The name of the CSV file to save the evaluation results.

        Returns:
            None

        Note:
            The method assumes that the SetFit model has been trained and is ready for evaluation.
            The 'val_df' parameter is a DataFrame containing the validation dataset with columns 'text' and 'label'.
            The evaluation metrics are computed based on the model's predictions on the validation data and the ground truth labels.
            The evaluation results are saved in a CSV file with the specified 'name' in the given directory.
        """
        filtered_df = tagged_df[tagged_df['verdict'] == file]
        y_true = filtered_df[label].values
            
        # y_pred = model.predict_proba(tagged_df['text'].values).numpy()[:, 1]
        y_pred = df_probabilities[column]
        y_pred_round = df_predictions[column]
        precision = precision_score(y_true
                                    ,y_pred_round)
        recall = recall_score(y_true, y_pred_round)
        f1 = f1_score(y_true, y_pred_round)
        precision_1, recall_1, thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = auc(recall_1, precision_1)
        # if logger is not None:
        #     logger.info(f"for label {label}: \n Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-PR: {auc_pr}")
        # else:
        #     print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-PR: {auc_pr}")
        # if label == 'reject':
        #     tagged_df = tagged_df.loc[tagged_df.index[y_pred_round] != 1]
        return precision, recall, f1, auc_pr
    

def load_classifires(eval_path, classifiers_path, logger, first_level_labels:list = None,
                          second_level_labels:list = None):

    classifiers = load_all_classifies(eval_path=eval_path,
                                        classifiers_path=classifiers_path,
                                        first_level_labels=first_level_labels,
                                        second_level_labels=second_level_labels,
                                        logger=logger)
    return classifiers, first_level_labels, second_level_labels


def convert_dataset_to_dataframe(datasets_dict):
    """
    Function that converts a defaultdict of datasets into a DataFrame with 
    sentences as one column and labels as binary columns.
    """
    data = []

    for label, dataset in datasets_dict.items():
        for example in dataset:
            text = example['text']
            label_value = example['label']
            existing_entry = next((item for item in data if item['text'] == text), None)
            if existing_entry:
                existing_entry[label] = label_value
            else:
                entry = {'text': text, label: label_value}
                data.append(entry)

    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)

    return df

