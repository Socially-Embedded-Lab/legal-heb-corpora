import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from datetime import datetime, timedelta

from setfit import SetFitModel, SetFitTrainer

# Determine the current directory and set the project path
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

# Import utility functions and custom modules
from utils.files import config_parser, setup_logger, write_yaml
from utils.sentence_classification import evaluate, load_datasets
from scripts.sentence_classification.setfit_handler import SetfitTrainerWrapper
from scripts.sentence_classification.setfit_version import SentenceTaggingModel

# Set the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(params,domain_path_model, tagger=False, date='', logger=None):
    """
    Train and evaluate models specified in the configuration, then save results and trained models.
    
    Args:
        params (dict): Configuration parameters for training and evaluation.
        domain_path (str): Path to the domain-specific directory for saving results.
        tagger (bool, optional): Indicates whether tagging is enabled. Defaults to False.
        date (str, optional): Custom date string for organizing results. Defaults to ''.
        logger (object, optional): Logger for recording process information. Defaults to None.
    """

    # Extract user-specific paths and parameters
    username = params['username']
    data_path = params['data_path'].format(username=username)
    save_dir = params['save_dir'].format(username=username)
    save_model_path = params['save_model_path'].format(username=username)

    # Log the start of training and the data source
    logger.info(f"\nTrain start, data is taken from {data_path}")

    # Iterate through each model configuration provided in the params
    for model_config in params['models_to_train']:
        model_name = ''
        if next(iter(model_config.values())):
            model_name = next(iter(model_config.keys()))
            # labels = params['labels']
            # logger.info(f'Starting {model_name} training for labels: {labels}')

        # Handle training for the 'setfit' model
        if model_name == 'setfit':
            # If generated data is provided, prepare datasets and train
            if params['generated_data']:
                # Load the pretrained SetFit model
                model = SetFitModel.from_pretrained(params["pretrained_model"]).to(device)

                # Iterate through data directories by label
                for label_name in os.listdir(data_path):
                    # Load train and validation datasets
                    train_df = pd.read_csv(os.path.join(data_path, label_name, label_name + '_train.csv'))
                    val_df = pd.read_csv(os.path.join(data_path, label_name, label_name + '_val.csv'))
                    
                    # Clean up unnecessary columns from the datasets
                    train_df = train_df.drop(['Type', 'Unnamed: 0'], axis=1)
                    val_df = val_df.drop(['Type', 'Unnamed: 0'], axis=1)
                    
                    # Rename columns to match the expected format
                    train_df.columns = ['text', 'label']
                    val_df.columns = ['text', 'label']
                    
                    # Split the validation data into test and validation sets
                    test_df, val_df = train_test_split(val_df, test_size=0.3, random_state=42)
                    
                    # Convert the DataFrames to datasets for SetFit training
                    train_ds, val_ds = load_datasets(train_df, val_df)

                    # Initialize the SetFit trainer
                    trainer = SetFitTrainer(
                        model=model.to(device),
                        train_dataset=train_ds,
                        eval_dataset=val_ds,
                        batch_size=16,
                        num_epochs=3
                    )
                    
                    # Clear CUDA cache to free memory
                    torch.cuda.empty_cache()
                    
                    # Train the model
                    trainer.train()

                    # Evaluate the model and save metrics
                    precision, recall, f1, auc_pr, model, val_df = evaluate(model, test_df)
                    metrics_path = os.path.join(save_dir, params['experiment_name'], f"{label_name}_metrics.txt")
                    with open(metrics_path, "w") as f:
                        f.write(f"Precision: {precision}\n")
                        f.write(f"Recall: {recall}\n")
                        f.write(f"F1 Score: {f1}\n")
                        f.write(f"AUC-PR: {auc_pr}\n")
                        f.write(f"Model: {model}\n")
                        f.write(f"Validation pos num DataFrame: {val_df['label'].value_counts().values[1]}\n")


            # Handle training with the SentenceTaggingModel
            elif params['setfit']:
                # Get the current date for file organization
                current_date = datetime.now() - timedelta(days=3)
                formatted_date = current_date.strftime("%m_%d")
                
                # Initialize the SentenceTaggingModel
                model = SentenceTaggingModel(data_path, save_dir)
                
                # Use a custom date if provided
                if len(date) != 0:
                    formatted_date = date

                # Set up directories for saving results and models
                save_dir = os.path.join(save_dir, formatted_date + '_' + params['experiment_name'])
                os.makedirs(save_dir, exist_ok=True)
                
                save_model_path = os.path.join(save_model_path,domain_path_model, formatted_date + '_' + params['experiment_name'])
                os.makedirs(save_model_path, exist_ok=True)
                
                # Save the configuration files for reproducibility
                write_yaml(os.path.join(save_dir, 'config.yaml'), params)
                write_yaml(os.path.join(save_model_path, 'config.yaml'), params)

                # Run the model training process
                model.run(
                    save_dir, params['save_model'], save_model_path, params['balance'], params['training_args'],
                    params['pretrained_model_list'], params['ignore_labels'], params['first_label_list'], params['second_label_list'],
                    logger
                )

            # Handle training with the SetfitTrainerWrapper
            else:
                st = SetfitTrainerWrapper(logger,
                                          train_path=data_path,
                                          save_dir=params["save_dir"],
                                          num_samples_list=params["num_samples_list"],
                                          model_name_initial=params["model_name_initial"],
                                          load_xlsx=params["load_xlsx"],
                                          all_class=params["all_class"],
                                          batch_size=params["batch_size"],
                                          num_iteration=params["num_iteration"],
                                          labels_=params["labels"],
                                          pretrained_model=params["pretrained_model"],
                                          pretrained_model_list=params["pretrained_model_list"],
                                          result_path=params['result_path'].format(username=username),
                                          generated_data=params['generated_data']
                                         )
                # Train and get the save path for results
                save_path = st.train(params['experiment_name'], params["load_xlsx"])
    # Log completion of training for the current model
    logger.info(f'{model_name} training finished.')


def main(params, domain_path_model):
    # Set up a logger to save logs in the specified directory and file
    logger = setup_logger(
        save_path=os.path.join(params['save_dir'].format(username=params['username']), 'logs'),
        file_name='unbelance_setnence_classification'
    )
    
    # Enable tagging for the classification process
    tagger = True

    # Run the sentence classification process with the given parameters and logger
    run(params=params,domain_path_model=domain_path_model, tagger=tagger, logger=logger)


if __name__ == "__main__":

    main_params = config_parser("", "main_config")    
    domain = main_params["domain"]    
    domain_path_model = f"{domain}"
    # Parse the training configuration parameters for the specific task
    domain = domain + "_sentence_cls"
    params = config_parser("", domain)
    
    main(params, domain_path_model)
