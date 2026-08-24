# Standard library imports
import os
import sys
import random
from collections import Counter

# Third-party library imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import roc_curve, roc_auc_score

# Set project directory for importing project-specific modules
current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.insert(0, pred_sentencing_path)

# Project-specific imports
from utils.files import convert_to_serializable, save_json, load_datasets

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, fbeta_score

def compute_metrics(predictions, labels, probabilities):
    """
    Compute various performance metrics for a classification task.
    
    Args:
        predictions (array-like): Predicted class labels.
        labels (array-like): True class labels.
        probabilities (array-like): Predicted probabilities for each class.
        
    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1 score, F2 score, PRAUC, and AUC.
    """
        
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    f2 = fbeta_score(labels, predictions, beta=2)  # F2 Score
    try:
        prauc = average_precision_score(labels, probabilities[:, 1])
        auc = roc_auc_score(labels, probabilities[:, 1])
    except:
        prauc = average_precision_score(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'PRAUC': prauc,
        'AUC': auc
    }


class SentenceTaggingModel:
    def __init__(self, file_path, save_path, seed=42, num_samples=8, num_eval_samples=100):
        """
        Initialize the class with file paths, configuration settings, and device setup.
        
        Args:
            file_path (str): Path to the input data file.
            save_path (str): Path to save the processed data or models.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            num_samples (int, optional): Number of samples to use for training. Defaults to 8.
            num_eval_samples (int, optional): Number of samples to use for evaluation. Defaults to 100.
        """
        # Initialize variables for the model and datasets
        self.model = None               # Placeholder for the model object
        self.train_dataset = None       # Placeholder for the training dataset
        self.eval_dataset = None        # Placeholder for the evaluation dataset
        self.test_dataset = None        # Placeholder for the testing dataset
        self.all_data_df = None         # Placeholder for the DataFrame holding all data

        # Set file path and save path
        self.file_path = file_path      # Path to the input file
        self.save_path = save_path      # Path where results or models will be saved

        # Set configuration parameters
        self.seed = seed                # Seed for reproducibility
        self.num_samples = num_samples  # Number of samples for training
        self.num_eval_samples = num_eval_samples  # Number of samples for evaluation

        # Set the computation device (use GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data if the file is an Excel file
        if self.file_path.endswith('.xlsx'):
            # Read the Excel file into a DataFrame using the openpyxl engine
            self.all_data_df = pd.read_excel(self.file_path, engine='openpyxl')


    def load_and_process_data(self, label):
        """
        Load and preprocess the data to create a balanced dataset for training.

        Args:
            label (str): The column name in the DataFrame representing the target label.

        Returns:
            Dataset: A Hugging Face Dataset object containing the balanced data.
        """
        # Select only the 'text' column and the target label column
        df_filtered = self.all_data_df[['text', label]]

        # Separate the data into majority and minority classes based on the label
        df_majority = df_filtered[df_filtered[label] == 0]  # Majority class
        df_minority = df_filtered[df_filtered[label] == 1]  # Minority class

        # Balance the dataset by undersampling or oversampling
        if len(df_majority) > len(df_minority):
            # Undersample the majority class to match the size of the minority class
            df_majority_undersampled = df_majority.sample(len(df_minority))
            df_balanced = pd.concat([df_majority_undersampled, df_minority])
        else:
            # Oversample the minority class to match the size of the majority class
            df_minority_oversampled = df_minority.sample(len(df_majority), replace=True)
            df_balanced = pd.concat([df_majority, df_minority_oversampled])

        # Ensure the 'text' column is of type string
        df_balanced['text'] = df_balanced['text'].astype(str)

        # Rename columns to standard names for compatibility with the Dataset object
        df_balanced.columns = ['sentence', 'label']

        # Convert the balanced DataFrame to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df_balanced)

        # Rename the auto-generated index column to 'idx' for clarity
        dataset = dataset.rename_column("__index_level_0__", "idx")

        # Map the dataset to ensure it has the correct structure
        dataset = dataset.map(lambda example: {
            "idx": example["idx"],
            "sentence": example["sentence"],
            "label": example["label"]
        })

        # Return the processed and balanced dataset
        return dataset


    def sample_balanced_eval(self, dataset, train_indices, label_column="label"):
        """
        Create a balanced evaluation dataset by sampling an equal number of instances 
        for each class, while ensuring the remaining data is used as the test dataset.

        Args:
            dataset (Dataset): The full dataset from which to sample.
            train_indices (list): Indices of the training dataset to exclude from evaluation and testing.
            label_column (str, optional): The column name representing the target label. Defaults to "label".

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the evaluation dataset and the test dataset.
        """
        # Exclude the training data to focus only on the remaining data for evaluation and testing
        remaining_indices = list(set(range(len(dataset))) - set(train_indices))
        remaining_dataset = dataset.select(remaining_indices)

        # Determine the number of samples to include in the evaluation dataset for each class
        label_counts = Counter(remaining_dataset[label_column])
        samples_per_class = self.num_eval_samples // len(label_counts)

        # Sample balanced data for the evaluation dataset by selecting equal instances for each class
        eval_indices = []
        for label in label_counts:
            label_indices = [i for i, x in enumerate(remaining_dataset[label_column]) if x == label]
            eval_indices.extend(random.sample(label_indices, min(samples_per_class, len(label_indices))))

        # Use the selected indices to create the evaluation dataset
        eval_dataset = remaining_dataset.select(eval_indices[:self.num_eval_samples])

        # Use the remaining data, after excluding evaluation samples, to create the test dataset
        remaining_indices = list(set(remaining_indices) - set(eval_indices))
        test_dataset = dataset.select(remaining_indices)

        # Return both the evaluation and test datasets
        return eval_dataset, test_dataset


    def sample_dataset_(self, dataset, label_column="label"):
        """
        Create a balanced subset of the dataset for training by sampling an equal number 
        of instances for each class.

        Args:
            dataset (Dataset): The full dataset from which to sample.
            label_column (str, optional): The column name representing the target label. Defaults to "label".

        Returns:
            Tuple[Dataset, list]: A tuple containing the sampled training dataset and the indices of the sampled data.
        """
        random.seed(self.seed)

        # Count the number of occurrences for each class in the dataset
        label_counts = Counter(dataset[label_column])

        # Initialize a list to store indices of the sampled training data
        train_indices = []

        # Sample an equal number of instances for each class
        for label in label_counts:
            label_indices = [i for i, x in enumerate(dataset[label_column]) if x == label]
            train_indices.extend(random.sample(label_indices, min(self.num_samples, len(label_indices))))

        # Create the training dataset from the sampled indices
        train_dataset = dataset.select(train_indices)

        # Return the training dataset and the sampled indices
        return train_dataset, train_indices


    def handle_overlaps(self, train_dataset, eval_dataset, test_dataset):
        """
        Handle and remove overlapping samples between train, evaluation, and test datasets.

        Args:
            train_dataset (Dataset): The training dataset.
            eval_dataset (Dataset): The evaluation dataset.
            test_dataset (Dataset): The test dataset.

        Returns:
            Tuple[Dataset, Dataset]: Updated evaluation and test datasets with overlaps removed.
        """

        def check_and_print_overlaps(train_set, eval_set, test_set):
            """
            Check for overlaps between the train, eval, and test datasets and print results.

            Args:
                train_set (set): Set of sentences from the training dataset.
                eval_set (set): Set of sentences from the evaluation dataset.
                test_set (set): Set of sentences from the test dataset.

            Returns:
                Tuple[set, set, set]: Overlaps between train-eval, train-test, and eval-test.
            """
            train_eval_overlap = train_set.intersection(eval_set)
            train_test_overlap = train_set.intersection(test_set)
            eval_test_overlap = eval_set.intersection(test_set)

            # Print the overlap details
            if not train_eval_overlap and not train_test_overlap and not eval_test_overlap:
                print("No overlap between the three datasets")
            else:
                if train_eval_overlap:
                    print("Overlap between train and eval:")
                    print(len(train_eval_overlap))
                if train_test_overlap:
                    print("Overlap between train and test:")
                    print(len(train_test_overlap))
                if eval_test_overlap:
                    print("Overlap between eval and test:")
                    print(len(eval_test_overlap))

            return train_eval_overlap, train_test_overlap, eval_test_overlap

        # Convert dataset 'text' columns into sets for overlap checking
        train_sentences = set(train_dataset['text'])
        eval_sentences = set(eval_dataset['text'])
        test_sentences = set(test_dataset['text'])

        # Identify overlaps between the datasets
        train_eval_overlap, train_test_overlap, eval_test_overlap = check_and_print_overlaps(train_sentences, eval_sentences, test_sentences)

        # Remove overlaps between training and evaluation datasets
        if train_eval_overlap:
            eval_dataset = eval_dataset.filter(lambda example: example['text'] not in train_eval_overlap)
            eval_sentences = set(eval_dataset['text'])

        # Remove overlaps between training and test datasets
        if train_test_overlap:
            test_dataset = test_dataset.filter(lambda example: example['text'] not in train_test_overlap)
            test_sentences = set(test_dataset['text'])

        # Remove overlaps between evaluation and test datasets
        if eval_test_overlap:
            test_dataset = test_dataset.filter(lambda example: example['text'] not in eval_test_overlap)
            test_sentences = set(test_dataset['text'])

        # Final check to confirm no overlaps remain
        check_and_print_overlaps(train_sentences, eval_sentences, test_sentences)

        # Return the updated evaluation and test datasets
        return eval_dataset, test_dataset



    def load_datasets(self, label, balance=False):
        """
        Load and prepare the train, evaluation, and test datasets, ensuring proper sampling and balancing.

        Args:
            label (str): The target column name in the dataset.
            balance (bool, optional): Whether to balance the dataset or not. Defaults to False.

        Returns:
            None: Updates the instance attributes `train_dataset`, `eval_dataset`, and `test_dataset`.
        """
        # Check if the data is already loaded into a DataFrame
        if self.all_data_df is not None:
            # Process the raw data into a Dataset object
            dataset = self.load_and_process_data(label)

            # Sample a balanced training dataset
            train_dataset, train_indices = self.sample_dataset_(dataset, label_column="label")

            # Sample evaluation and test datasets with balanced class representation
            eval_dataset, test_dataset = self.sample_balanced_eval(dataset, train_indices, label_column="label")

            # Remove overlaps between the train, evaluation, and test datasets
            eval_dataset, test_dataset = self.handle_overlaps(train_dataset, eval_dataset, test_dataset)

            # Handle edge case: If the test dataset is empty, split the evaluation dataset into two
            if len(test_dataset) == 0:
                num_eval_samples = len(eval_dataset)
                half_eval_samples = num_eval_samples // 2

                # Move half of the evaluation samples to the test dataset
                test_dataset = eval_dataset.select(range(half_eval_samples))
                eval_dataset = eval_dataset.select(range(half_eval_samples, num_eval_samples))

            # Shuffle datasets and remove unnecessary columns
            train_dataset = train_dataset.remove_columns(['idx']).shuffle(seed=self.seed)
            eval_dataset = eval_dataset.remove_columns(['idx']).shuffle(seed=self.seed)
            test_dataset = test_dataset.remove_columns(['idx']).shuffle(seed=self.seed)

        else:
            # If data is not preloaded, use an external function to load datasets
            train_dataset, eval_dataset, test_dataset = load_datasets(self.file_path, label, balance)

            # Remove overlaps between the train, evaluation, and test datasets
            eval_dataset, test_dataset = self.handle_overlaps(train_dataset, eval_dataset, test_dataset)

        # Update the instance attributes with the prepared datasets
        self.train_dataset, self.eval_dataset, self.test_dataset = train_dataset, eval_dataset, test_dataset


    def load_model(self, model_name="dicta-il/dictabert"):
        """
        Load and initialize the SetFit model with a differentiable head for classification.

        Args:
            model_name (str, optional): The name of the pretrained model to load. Defaults to "dicta-il/dictabert".

        Returns:
            None: The loaded model is stored in the instance attribute `self.model`.
        """
        # Load a pretrained SetFit model with a custom head for classification
        self.model = SetFitModel.from_pretrained(
            model_name,
            use_differentiable_head=True,  # Use a differentiable head for the model
            head_params={"out_features": 2}  # Set the number of output features (e.g., binary classification)
        )

        # Move the model to the appropriate device (CPU or GPU)
        self.model.to(self.device)


    def train_model(self, training_args):
        """
        Train the SetFit model using specified training arguments.

        Args:
            training_args (dict): A dictionary of training configuration parameters.

        Returns:
            None: The model is trained and updated in place.
        """
        # Define training arguments based on the provided configuration
        args = TrainingArguments(
            batch_size=training_args['batch_size'],                  # Batch size for training
            body_learning_rate=tuple(training_args['body_learning_rate']),  # Learning rate for the model's body
            head_learning_rate=training_args['head_learning_rate'],  # Learning rate for the classification head
            l2_weight=training_args['l2_weight'],                    # L2 regularization weight
            num_epochs=training_args['num_epochs'],                  # Number of training epochs
            end_to_end=training_args['end_to_end'],                  # Whether to fine-tune the entire model
            evaluation_strategy=training_args['evaluation_strategy'],  # Evaluation strategy (e.g., steps or epoch)
            save_strategy=training_args['save_strategy'],            # Save strategy for checkpoints
            eval_steps=training_args['eval_steps'],                  # Number of steps between evaluations
            load_best_model_at_end=training_args['load_best_model_at_end'],  # Load the best model at the end of training
        )

        # Initialize the Trainer with the model, datasets, and configuration
        trainer = Trainer(
            model=self.model,                         # The pretrained model to be trained
            args=args,                                # Training arguments
            train_dataset=self.train_dataset,         # Training dataset
            eval_dataset=self.eval_dataset,           # Evaluation dataset
            column_mapping={"text": "text", "label": "label"},  # Map dataset columns to expected inputs
            metric="f1",                              # Metric to evaluate during training
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Use early stopping to avoid overfitting
        )

        # Train the model using the specified trainer
        trainer.train()



            
    def find_best_threshold(self, labels, probabilities):
        """
        Determine the best decision threshold for a classification model based on ROC curve,
        selecting the threshold that maximizes Youden's J statistic, and return the ROC AUC.

        Args:
            labels (array-like): True binary labels for the dataset.
            probabilities (array-like): Predicted probabilities for each class, typically from the model.
                                        It is assumed probabilities[:, 1] corresponds to the positive class.

        Returns:
            tuple: A tuple containing:
                - float: The threshold that maximizes Youden's J (sensitivity - FPR).
                - float: The ROC AUC score.
        """
        # Compute the ROC curve: fpr, tpr, and thresholds.
        fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
        
        # Calculate Youden's J statistic for each threshold.
        J = tpr - fpr
        
        # Find the index of the threshold that maximizes J.
        best_idx = np.argmax(J)
        
        # Retrieve the corresponding best threshold.
        best_threshold = thresholds[best_idx]
        
        
        return best_threshold

    
    def evaluate_model(self, save_path):
        """
        Evaluate the model on the test dataset, compute metrics, and generate a confusion matrix.

        Args:
            save_path (str): Directory path where evaluation outputs (e.g., confusion matrix) are saved.

        Returns:
            Tuple[dict, pd.DataFrame]: A dictionary of evaluation metrics and a DataFrame of misclassified examples.
        """
        # Extract sentences and labels from the test dataset
        sentences = self.test_dataset['text']
        labels = self.test_dataset['label']

        # Get predicted probabilities for the sentences
        with torch.no_grad():
            probabilities = self.model.predict_proba(sentences)
        try:
            # Ensure probabilities are in a NumPy array format (handle potential Tensor output)
            probabilities = probabilities.numpy()
        except:
            probabilities = probabilities.cpu().detach().numpy()

        # Find the best threshold for classification based on F1 score
        best_threshold = self.find_best_threshold(labels, probabilities)

        # Generate binary predictions using the best threshold
        predictions = (probabilities[:, 1] >= best_threshold).astype(int)

        # Compute evaluation metrics (accuracy, precision, recall, F1, etc.)
        metrics = compute_metrics(predictions, labels, probabilities)
        metrics['best_threshold'] = best_threshold

        # Generate a confusion matrix and visualize it
        conf_matrix = confusion_matrix(labels, predictions)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        # Save the confusion matrix plot to the specified path
        conf_matrix_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.show()
        plt.clf()  # Clear the figure to avoid overlapping plots

        plt.close()  # Close the figure to free resources

        # Identify misclassified examples for analysis
        predictions = np.array(predictions)
        labels = np.array(labels)
        errors = (predictions != labels)
        error_df = pd.DataFrame({
            'sentence': sentences,                          # Original text
            'true_label': labels,                          # Ground-truth labels
            'predicted_label': predictions,                # Model predictions
            'probabilities': [prob[1] for prob in probabilities],  # Probability of the positive class
        })
        # Filter and sort misclassified examples by confidence level (descending)
        error_df = error_df[errors].sort_values(by='probabilities', ascending=False)

        # Return the evaluation metrics and the DataFrame of misclassified examples
        return metrics, error_df

    
    def save_results(self, save_path, metrics, error_df):
        """
        Save evaluation metrics and misclassified examples to files.

        Args:
            save_path (str): Directory path where results will be saved.
            metrics (dict): Dictionary of evaluation metrics.
            error_df (pd.DataFrame): DataFrame containing misclassified examples.

        Returns:
            None
        """
        # Save the metrics as a JSON file for easy interpretation and sharing
        save_json(metrics, os.path.join(save_path, 'metric.json'))

        # Save the misclassified examples as a CSV file for detailed analysis
        error_df.to_csv(os.path.join(save_path, 'error_df.csv'))

        
    def save_model(self, model_path):
        """
        Save the trained model to the specified directory.

        Args:
            model_path (str): Directory path where the model will be saved.

        Returns:
            None
        """
        # Define the path for saving the pretrained model
        model_path = os.path.join(model_path, 'model')

        # Save the model to the specified path
        self.model.save_pretrained(model_path)

        # Print confirmation of the model saving process
        print(f"Model saved at {model_path}")

    
    

    def run(self, save_dir, save_model, save_model_path, balance, training_args, pretrained_model_list,
            ignore_labels, first_label_list=[], second_label_list=[], logger=None):
        """
        Run the full training, evaluation, and model saving process for a set of labels.

        Args:
            save_dir (str): Directory where results will be saved.
            save_model (bool): Whether to save the trained model.
            save_model_path (str): Directory where the model will be saved.
            balance (bool): Whether to balance the datasets.
            training_args (dict): Configuration parameters for training.
            pretrained_model_list (list): List of pretrained models to train.
            ignore_labels (list): List of labels to skip.
            first_label_list (list, optional): Primary list of labels to train on. Defaults to [].
            second_label_list (list, optional): Secondary list of labels to train on. Defaults to [].
            logger (object, optional): Logger to record the process. Defaults to None.

        Returns:
            None
        """
        # Clear CUDA cache to free memory before starting
        torch.cuda.empty_cache()

        # Merge first and second label lists, if provided
        if first_label_list is None:
            label_list = second_label_list
        elif second_label_list is None:
            label_list = first_label_list
        else:
            label_list = first_label_list + second_label_list

        # Iterate through each label in the list
        for label in label_list:
            # Skip labels in the ignore list
            if label in ignore_labels:
                continue
            if label in first_label_list:
                kind_of_label = 'first_labels'
            elif label in second_label_list:
                kind_of_label = 'second_labels'

            
            # Log the start of training for the current label
            logger.info(f'Starting training for label: {label}')

            # Create a directory for saving results for the current label
            save_path = os.path.join(save_dir, kind_of_label, label)
            os.makedirs(save_path, exist_ok=True)

            #Create a directory for saving the model for the current label
            save_path_model = os.path.join(save_model_path, kind_of_label, label)
            os.makedirs(save_model_path, exist_ok=True)

            try:
                # Load and prepare datasets for the current label
                self.load_datasets(label, balance)

                if pretrained_model_list is not None:
                    # Train multiple pretrained language models
                    for pretrained_model in pretrained_model_list:
                        model_name = pretrained_model.split('/')[-1]
                        torch.cuda.empty_cache()

                        logger.info(f'Starting training for model: {model_name}')

                        # Load the pretrained model
                        self.load_model(pretrained_model)

                        # Train the model with specified arguments
                        self.train_model(training_args)

                        # Save evaluation results and metrics
                        model_save_path = os.path.join(save_path, model_name)
                        os.makedirs(model_save_path, exist_ok=True)
                        metrics, error_df = self.evaluate_model(model_save_path)
                        metrics = convert_to_serializable(metrics)

                        logger.info(f'Metrics for {label} with model {model_name}: {metrics}')
                        self.save_results(model_save_path, metrics, error_df)

                        # Save the trained model if required
                        if save_model:
                            save_model_path__ = os.path.join(save_path_model, model_name)
                            self.save_model(save_model_path__)

                        # Clear CUDA cache to free memory
                        torch.cuda.empty_cache()

                else:
                    # Train with a default pretrained model if no list is provided
                    self.load_model()
                    self.train_model(training_args)
                    metrics, error_df = self.evaluate_model(save_path)
                    metrics = convert_to_serializable(metrics)

                    logger.info(f'Metrics for {label}: {metrics}')
                    self.save_results(save_path, metrics, error_df)

                    # Save the trained model if required
                    if save_model:
                        self.save_model(save_path)

                # Clear CUDA cache to free memory
                torch.cuda.empty_cache()

            except Exception as e:
                # Log any errors encountered during processing
                logger.error(f'Error in label {label}: {e}')
                torch.cuda.empty_cache()


        return
