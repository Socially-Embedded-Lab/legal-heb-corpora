import os
import sys
import random
import pandas as pd

current_dir = os.path.abspath(__file__)
pred_sentencing_path = os.path.abspath(os.path.join(current_dir, '..', '..','..','..'))
sys.path.insert(0, pred_sentencing_path)


from src_old.old.preprocess.Preprocessing_flows import *
from src_old.old.Utils.Loads import  extract_config_params

def shuffle_array(arr):
    """
        Shuffle the elements of a list randomly using the Fisher-Yates (Knuth) shuffle algorithm.
    """
    n = len(arr)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

# TODO change the preprocessing_flow
class data_handler:

    def __init__(self, data_path: str = None,
                 positive_number: int = None,
                 labels_: list = None,
                 load_xlsx: bool = None,
                 single_df: bool = True,
                 SEED: int = 7):
        """
            Initialize the object for data preprocessing and loading.

            This constructor method initializes the object used for data preprocessing and loading.
            It allows for two modes of data loading:
            - If 'single_df' is True, the provided CSV file ('data_path') is assumed to contain multiple columns, where each column corresponds to a different label.
              Separate DataFrames are created for each label, and they are stored in the 'dfs' dictionary, with the 'text' and 'label' columns for each label DataFrame.
              The 'dfs' dictionary also contains a 'case' DataFrame extracted from the 'case' column of the same CSV file.
            - If 'single_df' is False, the data is preprocessed and loaded using the 'preprocessing_flow' function from another module.
              The resulting main DataFrame is stored as 'df', and separate DataFrames for each label are stored in the 'dfs' dictionary.

            Parameters:
                data_path (str, optional): The file path of the CSV or Excel file to load the data from. Defaults to None.
                positive_number (int, optional): The numeric representation of the positive label. Defaults to None.
                labels_ (list, optional): The list of labels to consider. Defaults to None.
                load_xlsx (bool, optional): Whether the data should be loaded from an Excel file instead of a CSV file. Defaults to None.
                single_df (bool, optional): Whether the data should be loaded as a single DataFrame or preprocessed into multiple DataFrames. Defaults to True.
                SEED (int, optional): The random seed used for data preprocessing. Defaults to 7.

            Returns:
                None

            Note:
                The behavior of the object depends on the 'single_df' parameter.
                If 'single_df' is True, the 'data_path' is assumed to be a CSV file, and the 'labels_' parameter must be provided.
                If 'single_df' is False, the 'preprocessing_flow' function is used, and additional preprocessing settings are considered.
            """

        self.data_path = data_path
        self.positive_number = positive_number
        self.labels_ = labels_
        if single_df:
            self.dfs = {}
            if os.path.exists(data_path):
                if load_xlsx:
                    df = pd.read_excel(data_path)
                else:    
                    df = pd.read_csv(data_path)
                if 'Label' in df.columns:
                    one_hot_df = pd.get_dummies(df.Label)
                    one_hot_df.drop(columns=['Unnamed: 0'], inplace=True)
                    df = pd.concat([df, one_hot_df], axis=1)

                df.rename(columns={
                            'verdict': 'Case',
                            'text': 'Text'
                            }, inplace=True)
                
                for label in self.labels_:
                    label = label.replace("'", '')
                    data = {'text': df['Text'].values, 'label': df[label].values}
                    self.dfs[label] = pd.DataFrame(data)
                self.dfs['case'] = df['Case'].values
        else:
            self.df, self.dfs, multy_label_dict = preprocessing_flow(data_path, SEED, load_xlsx=load_xlsx,
                                                                     labels=labels_)

    def shuffle(self, label):
        """
        Shuffle and create a DataFrame with balanced samples for a specific label.

        This method takes a specific 'label' and shuffles the data associated with that label and the data from other labels.
        It then creates a DataFrame with balanced samples, containing an equal number of positive and negative instances.
        The number of positive instances is controlled by the 'positive_number' parameter specified during object initialization.

        Parameters:
            label (str): The label for which the data needs to be shuffled and balanced.

        Returns:
            pd.DataFrame: A DataFrame with balanced samples, containing an equal number of positive (label=1) and negative (label=0) instances.

        Note:
            The method relies on the 'df' DataFrame, which should have been initialized during object creation.
            The 'positive_number' parameter, specified during object initialization, controls the number of positive instances to include in the DataFrame.
            The DataFrame returned contains equal-sized samples of the provided 'label' and other labels to achieve a balanced dataset.
        """
        positives = []
        negatives = []

        for row in self.df.iterrows():
            row = row[1]
            if row['label'] == label:
                positives.append(row['text'])
            else:
                negatives.append(row['text'])

        shuffle_array(positives)
        shuffle_array(negatives)

        data = {'text': [], 'label': []}
        positive_iter = 0
        negative_iter = 0

        while positive_iter < len(positives):
            for i in range(self.positive_number):
                if positive_iter >= len(positives):
                    break
                data['text'].append(positives[positive_iter])
                data['label'].append(1)
                positive_iter += 1

            for j in range(self.positive_number):
                if i < self.positive_number - 1:
                    for k in range(i):
                        data['text'].append(negatives[negative_iter])
                        data['label'].append(0)
                        negative_iter += 1
                    break

                data['text'].append(negatives[negative_iter])
                data['label'].append(0)
                negative_iter += 1

        return pd.DataFrame(data)

    def create_dict_labels(self):
        """
        Create a dictionary of DataFrames with equal-sized samples for each label.

        This method generates a dictionary of DataFrames, where each DataFrame corresponds to a specific label from the 'labels_' list.
        The DataFrames in the dictionary are created by shuffling the data corresponding to each label using the 'shuffle' method.
        The 'shuffle' method ensures that each DataFrame contains an equal number of samples for each label, helping to balance the data.

        Parameters:
            None

        Returns:
            dict: A dictionary of DataFrames, where each key is a label from the 'labels_' list, and each value is the corresponding DataFrame.

        Note:
            The 'shuffle' method is called for each label to create DataFrames with equal-sized samples for each label.
            The 'labels_' parameter must be provided during object initialization.
        """
        df_dict = {}
        for label in self.labels_:
            df = self.shuffle(label)
            df_dict[label] = df

        return df_dict
        df_dict = {}
        for label in self.labels_:
            df = self.shuffle(label)
            df_dict[label] = df

        return df_dict



def run():

    # seed, train_file_path, test_file_path, save_directory, verdict_path, num_samples_list, model_name_initial, \
    # param, all_class, batch_size, num_iteration, num_epoch, labels_, pretrained_models = extract_config_params(config)

    train_file_path = 'C:/Users/max_b/PycharmProjects/moj_sl/pred-sentencing/resources/data/new_data/train_test/second_lvl/test.csv'
    labels_ = ['reject', 'CONFESSION', 'CIRCUM_OFFENSE', 'GENERAL_CIRCUM', 'PUNISHMENT']

    handler = data_handler(data_path=train_file_path,
                           positive_number=8,
                           labels_=labels_,
                           SEED=42)

    df_dict = handler.create_dict_labels()



if __name__ == '__main__':
    run()