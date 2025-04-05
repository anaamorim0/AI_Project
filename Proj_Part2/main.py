import pandas as pd
import numpy as np
from argparse import ArgumentParser
from decisionTree import DecisionTree

def preprocess_connect4_dataset(filename):
    column_names = [f'col_{i}' for i in range(1, 43)] + ['Class']
    data = pd.read_csv(filename, header=None, names=column_names)
    data.insert(0, 'ID', range(1, len(data) + 1))
    return data

def discretize_numeric_values(dataset):
    for col in dataset.select_dtypes(include=[np.number]).columns:
        dataset[col] = pd.cut(dataset[col], bins=3, labels=['low', 'medium', 'high'])
    return dataset

def main():
    parser = ArgumentParser(description='Decision Tree')
    parser.add_argument('-tr', '--train', required=True, help='Dataset que vai ser usado para treinar a Decision Tree')
    parser.add_argument('-t', '--test', help='Dataset que vai ser usado para testar a Decision Tree')

    args = parser.parse_args()

    try:
        dataset_path = args.train
        dataset = pd.read_csv(dataset_path, na_values=['NaN'], keep_default_na=False)
        if args.train == 'connect4.csv':
            dataset = preprocess_connect4_dataset(dataset_path)
        if 'ID' in dataset.columns:
            dataset = dataset.drop(columns=['ID'])
        dataset = discretize_numeric_values(dataset)
        print("A treinar o classificador da decision tree no '" + str(args.train) + "' dataset:\n")
        tree = DecisionTree(dataset)
        print(tree)
    except FileNotFoundError:
        print("O dataset '" + str(args.train) + "' não existe.")
        exit()

    if args.test:
        try:
            test_path = args.test
            test_dataset = pd.read_csv(test_path, na_values=['NaN'], keep_default_na=False)
            if 'ID' in test_dataset.columns:
                test_dataset = test_dataset.drop(columns=['ID'])
            print("A prever os valores para o'" + str(args.test) + "' dataset:\n")
            pred = tree.predict(test_dataset)
            print(pred)

            true_labels = test_dataset[test_dataset.columns[-1]].tolist()

            accuracy = DecisionTree.accuracy(true_labels, predictions)
            print(f"Accuracy: {accuracy:.2f}")


        except FileNotFoundError:
            print("O dataset '" + str(args.test) + "' não esxite.")

if __name__ == "__main__":
    main()
