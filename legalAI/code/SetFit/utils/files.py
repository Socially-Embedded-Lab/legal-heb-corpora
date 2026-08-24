from collections import defaultdict
import pickle
import csv
import json
import os
from docx import Document
import numpy as np
import pandas as pd
import yaml
import logging
from datetime import datetime
from colorama import Fore, Style
from datasets import Dataset
from datetime import datetime


def get_date():
    """
    Get the current date in DD_MM format.

    Returns:
        str: The current date as a string in the format 'DD_MM'.
    """
    current_date = datetime.now()
    day = f'{current_date.day:02}'  # Zero-padded day
    month = f'{current_date.month:02}'  # Zero-padded month
    return f'{day}_{month}'


def read_docx(docx_path):
    """
    Read the content of a DOCX file and return the entire text.

    Args:
        docx_path (str): The path to the DOCX file.

    Returns:
        str: The entire text content of the document.
    """
    try:
        doc = Document(docx_path)
        text_content = ''

        # Iterate over all paragraphs and concatenate their text
        for paragraph in doc.paragraphs:
            text_content += paragraph.text + '\n'

        return text_content.strip()  # Remove trailing newline

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    
def load_json(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_pkl(path):
    """
    Load a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        object: The content of the pickle file.
    """
    with open(path, 'rb') as file:
        content = pickle.load(file)
    return content


def save_json(content, path):
    """
    Save data to a JSON file.

    Args:
        content (dict): Data to save.
        path (str): Path to save the JSON file.
    """
    with open(path, "w") as f:
        json.dump(content, f)


def create_csv(file_path, headers):
    """
    Create a CSV file with the specified headers.

    Args:
        file_path (str): Path to the CSV file.
        headers (list): List of column headers for the CSV file.
    """
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the headers
        writer.writerow(headers)


def append_to_csv(file_path, data):
    """
    Append data to an existing CSV file.

    Args:
        file_path (str): Path to the CSV file.
        data (list): Data to append as a row.
    """
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the data row
        writer.writerow(data)

        
def flatten_dict(d):
    """
    Flatten a nested dictionary into a single-level dictionary.

    Args:
        d (dict): A nested dictionary.

    Returns:
        dict: A flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.update(flatten_dict(v))
        else:
            items[k] = v
    return items


def config_parser(func='', config_name=''):
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        func (str): Subdirectory under the 'configs' folder indicating the function or context.
        config_name (str): Name of the YAML file (without the extension).

    Returns:
        dict: Dictionary representation of the YAML file contents.
    """
    if config_name == '':
        # Construct the absolute path to the configuration file
        config_path = os.path.join(
            os.path.abspath(__file__).split('src')[0],  # Get the base directory
            'resources/configs', func + '.yaml'  # Append the relative path
        )
    else:
        # Construct the absolute path to the configuration file
        config_path = os.path.join(
            os.path.abspath(__file__).split('src')[0],  # Get the base directory
            'resources/configs', func, config_name + '.yaml'  # Append the relative path
        )
    
    # Load the YAML file and return its contents
    with open(config_path, 'r') as file:
        data = yaml.safe_load(file)
        return data

    
def yaml_load(path):
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Dictionary representation of the YAML file contents.
    """
    # Open the YAML file at the specified path
    with open(path, 'r') as file:
        # Parse the YAML file contents into a Python dictionary
        data = yaml.safe_load(file)
        return data
    

def setup_logger(save_path, file_name):
    """
    Set up a logger that writes log messages to both a file and the console, with optional colored output.

    Args:
        save_path (str): Directory where the log file will be saved.
        file_name (str): Base name of the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger object with the current module's name
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG for detailed output

    # Get the current date to include in the log file name
    current_date_time = datetime.now()
    current_date = current_date_time.date()

    # Define the log file path and name
    log_file = os.path.join(save_path, f'{current_date}_{file_name}.txt')

    # Create a file handler to write log messages to the file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to output log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define a log message format for both file and console handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Define a class for adding colored output to the console handler
    class ColoredFormatter(logging.Formatter):
        """
        Custom formatter to add color to console log output based on log level.
        """
        def format(self, record):
            log_str = super().format(record)  # Format the log message
            dash_line = '-' * 80  # Add a separator line for visual clarity
            if record.levelname == 'DEBUG':
                return f'{Fore.GREEN}{dash_line}\n{log_str}\n{dash_line}{Style.RESET_ALL}'
            elif record.levelname == 'WARNING':
                return f'{Fore.YELLOW}{dash_line}\n{log_str}\n{dash_line}{Style.RESET_ALL}'
            elif record.levelname == 'ERROR':
                return f'{Fore.RED}{dash_line}\n{log_str}\n{dash_line}{Style.RESET_ALL}'
            elif record.levelname == 'CRITICAL':
                return f'{Style.BRIGHT}{Fore.RED}{dash_line}\n{log_str}\n{dash_line}{Style.RESET_ALL}'
            else:
                return f'{dash_line}\n{log_str}\n{dash_line}'

    # Apply the colored formatter to the console handler
    console_handler.setFormatter(
        ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    )

    # Return the configured logger
    return logger


def weap_type_extract(tagged_df):
    """
    Extract weapon type information from a tagged DataFrame.

    Args:
        tagged_df (pd.DataFrame): DataFrame containing tagged features, including weapon type columns.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'TYPE_WEP' that aggregates weapon type information.
    """
    # Initialize a list to store the aggregated weapon type information for each row
    new_coulmn = []

    # Iterate through each row in the DataFrame
    for _, row in tagged_df.iterrows():
        new_cell = []  # List to collect weapon types for the current row

        # Iterate through each column to find weapon type information
        for column in tagged_df.columns:
            if "WEP_TYPE" in column:  # Check if the column contains weapon type information
                if not pd.isna(row[column]) and row[column] != "":
                    # Add the weapon type (after removing the prefix) to the current row's list
                    new_cell.append(column.replace('WEP_TYPE-', ''))

        # Append the aggregated weapon types for the current row to the new column
        new_coulmn.append(new_cell)

    # Add the new column with aggregated weapon types to the DataFrame
    tagged_df['TYPE_WEP'] = new_coulmn

    # Return the updated DataFrame
    return tagged_df
        
        
def write_yaml(save_path, content):
    """
    Save content to a YAML file.

    Args:
        save_path (str): Path where the YAML file will be saved.
        content (dict): Data to be saved in YAML format.

    Returns:
        None
    """
    # Open the specified file path and write the content as YAML
    with open(save_path, 'w') as file:
        yaml.dump(content, file)


def reformat_sentence_tagged_file(data_path):
    """
    Convert a tagged CSV or Excel file into a reformatted DataFrame with 'text' and 'label' columns.

    Args:
        data_path (str): The file path of the tagged CSV or Excel file.

    Returns:
        pd.DataFrame: A new DataFrame with reformatted rows containing 'text' and 'label' columns,
                      and optionally a 'verdict' column for Excel files.
    """
    # Load the data based on the file type (CSV or Excel)
    if 'xlsx' not in data_path:
        df = pd.read_csv(data_path)  # Load CSV file
    else:
        df = pd.read_excel(data_path)  # Load Excel file

    # Initialize a list to store reformatted data
    data = []

    # Iterate through each row of the DataFrame
    for _, row in df.iterrows():
        text = row['text']  # Extract the 'text' column from the current row

        # Process each column in the row
        for column, value in row.items():
            if 'xlsx' not in data_path:
                # For CSV files: Append relevant rows with specific label conditions
                if column.lower() not in ['text', 'verdict', 'reject'] and \
                   'Unnamed' not in column and value == 1:
                    data.append({'text': text, 'label': column})
            else:
                # For Excel files: Append rows with additional 'verdict' information
                if column.lower() != 'text' and value == 1 and column.lower() != 'reject':
                    verdict = row['verdict'].split("_tagged")[0]  # Extract the base 'verdict' value
                    data.append({'text': text, 'label': column, 'verdict': verdict})

    # Create a new DataFrame with the reformatted data
    new_df = pd.DataFrame(data)
    return new_df


def aggrigate_sentence_cls_xlsx(path_to_files):
    """
    Aggregate multiple Excel files from a directory into a single combined Excel file.

    Args:
        path_to_files (str): Directory path containing the Excel files to aggregate.

    Returns:
        None: The combined Excel file is saved to the same directory.
    """
    # Initialize a list to store individual DataFrames
    dataframes = []

    # Iterate through all files in the specified directory
    for file_name in os.listdir(path_to_files):
        # Read the Excel file into a DataFrame
        temp_df = pd.read_excel(os.path.join(path_to_files, file_name))

        # Remove columns with names starting with 'Unnamed:' (likely auto-generated index columns)
        temp_df = temp_df.loc[:, ~temp_df.columns.str.startswith('Unnamed:')]

        # Append the cleaned DataFrame to the list
        dataframes.append(temp_df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Define the output file path for the combined Excel file
    output_path = os.path.join(path_to_files, 'combain_tagged_gt.xlsx')

    # Save the combined DataFrame to an Excel file
    combined_df.to_excel(output_path, index=False)

    # Print a confirmation message with the output file path
    print(f'\n Saved combined Excel file to {output_path}!')


def aggrigate_sentence_cls(path2v1, path2v2, save_path):
    """
    Aggregate sentence generation CSV files between two versions by aligning similar labels.

    Args:
        path2v1 (str): Directory path containing the first set of CSV files.
        path2v2 (str): Directory path containing the second set of CSV files.
        save_path (str): Directory path where the combined CSV files will be saved.

    Returns:
        None: Combined CSV files are saved to the specified directory.
    """
    # Iterate over all files in the first directory (v1)
    for file1 in os.listdir(path2v1):
        if file1 in os.listdir(path2v2):  # Check if the file also exists in v2
            label_name = file1.split('_generated_sentences.csv')[0]  # Extract label name
            v1_df = pd.read_csv(os.path.join(path2v1, file1))  # Load the v1 CSV file
            v2_df = pd.read_csv(os.path.join(path2v2, file1))  # Load the v2 CSV file

            # Remove duplicate rows from both DataFrames
            v1_df = v1_df.drop_duplicates()
            v2_df = v2_df.drop_duplicates()

            # Initialize an empty DataFrame with the same columns as v1_df
            combined_df = pd.DataFrame(columns=v1_df.columns)

            # Iterate through rows of v1 DataFrame
            for i, row_v1 in v1_df.iterrows():
                # Add the current v1 row to the combined DataFrame
                combined_df = pd.concat([combined_df, v1_df.iloc[i:i + 1]])

                # Skip further processing if the row is not an 'original' sentence
                if row_v1['Type'] != 'original':
                    continue

                # It's an original sentence; search for similar sentences in v2
                target_sentence = row_v1['Sentence']
                add_row = False

                # Iterate through rows of v2 DataFrame
                for j, row_v2 in v2_df.iterrows():
                    if add_row:
                        # Add v2 rows following the matching sentence, but stop if another 'original' is encountered
                        if row_v2['Type'] == 'original':
                            break
                        combined_df = pd.concat([combined_df, v2_df.iloc[j:j + 1]])
                        continue

                    # Mark the matching sentence and prepare to add subsequent rows
                    if row_v2['Sentence'] == target_sentence:
                        add_row = True
                        continue

            # Save the combined DataFrame to a CSV file
            save_file_path = os.path.join(save_path, f'{label_name}_generated_sentences.csv')
            combined_df.to_csv(save_file_path, index=False)
            print(f'Saved combined {label_name} DF to {save_file_path}!')

                
def filter_none_rows(dataset, column_name):
    """
    Filters out rows from the dataset where the specified column contains None values.
    """
    return dataset.filter(lambda example: example[column_name] is not None)


def load_datasets(data_path, label, balance):
    """
    Load datasets for training, evaluation, and testing from pickled files.

    Args:
        data_path (str): Path to the directory containing the dataset pickle files.
        label (str): Target label for the datasets.
        balance (bool): Whether to load balanced training datasets.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training, evaluation, and testing datasets.
    """
    def load_dataset_from_pkl(file_name):
        """
        Load a dataset from a pickle file, filter rows with valid labels, and return it as a Dataset.

        Args:
            file_name (str): Name of the pickle file to load.

        Returns:
            Dataset: Processed dataset loaded from the pickle file.
        """
        # Load the pickle file and extract the DataFrame for the specified label
        df = load_pkl(os.path.join(data_path, file_name))[label]

        # Convert the DataFrame to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        # Remove auto-generated index column if present
        if '__index_level_0__' in dataset.column_names:
            dataset = dataset.remove_columns(['__index_level_0__'])

        # Filter rows where the 'label' column has valid (non-null) values
        dataset = filter_none_rows(dataset, 'label')

        return dataset

    # Load the training dataset, selecting either balanced or regular based on the flag
    if balance:
        train_dataset = load_dataset_from_pkl('train_balance_label_dataframes.pkl')
    else:
        train_dataset = load_dataset_from_pkl('train_label_dataframes.pkl')

    # Load evaluation and testing datasets
    eval_dataset = load_dataset_from_pkl('eval_label_dataframes.pkl')
    test_dataset = load_dataset_from_pkl('test_label_dataframes.pkl')

    # Return all three datasets
    return train_dataset, eval_dataset, test_dataset


def convert_to_serializable(obj):
    """
    Recursively convert objects into JSON-serializable formats.

    Args:
        obj (any): The object to be converted.

    Returns:
        any: A JSON-serializable version of the input object.
    """
    if isinstance(obj, dict):
        # If the object is a dictionary, recursively process its keys and values
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # If the object is a list, recursively process its elements
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.generic):
        # If the object is a NumPy generic type, convert it to a Python native type
        return obj.item()
    else:
        # For other types, return the object as-is
        return obj


def load_all_datasets(data_path, second_level_labels):
    """
    Load multiple datasets for a set of second-level labels.

    Args:
        data_path (str): Path to the directory containing the dataset files.
        second_level_labels (list): List of second-level labels to load datasets for.

    Returns:
        defaultdict: A dictionary where keys are labels and values are the corresponding test datasets.
    """
    # Initialize a dictionary to store datasets, defaulting to an empty Dataset if not found
    testset_dict = defaultdict(Dataset)

    # Iterate through each label in the provided list
    for label in second_level_labels:
        try:
            # Attempt to load the test dataset for the current label
            _, _, test_set = load_datasets(data_path, label, balance=True)
            testset_dict[label] = test_set  # Store the test dataset in the dictionary
        except:
            # Print the label if there is an error loading its dataset
            print(f'\t\n{label}\n')

    # Return the dictionary containing all loaded test datasets
    return testset_dict



        