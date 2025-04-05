from math import log2
from collections import Counter
import numpy as np


class DecisionTree:

    def __init__(self, dataset):
        dataset = self.bool_to_string(dataset)
        attributes = dataset.columns.tolist()[1:-1]
        self.tree = self.id3(dataset, dataset, attributes)


    def predict(self, dataset):
        pred = dataset.apply(lambda row: self.tracing(self.tree, row), axis=1)
        pred = pred.apply(lambda pred: pred[0] if pred is not None else pred)
        
        return pred.tolist()


    def best_split_val(self, data, attribute):
        attr = data[attribute].unique()
        split_val = None
        best_information_gain = float('-inf')

        if len(attr) == 1:
            return attr[0]

        for val in attr:
            subset1 = data[data[attribute] <= val]
            subset2 = data[data[attribute] > val]

            target1 = subset1[data.columns[-1]].tolist()
            target2 = subset2[data.columns[-1]].tolist()

            information_gain = self.gain(data[data.columns[-1]].tolist(), target1, target2)

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                split_val = val

        return split_val

    def tracing(self, tree, row):
        attr = next(iter(tree))
        val = row[attr]

        if isinstance(val, bool):
            val = str(val)

        if isinstance(tree[attr], dict):
            split_key, split_val = next(iter(tree[attr])).split(' ')
            try:
                numeric_val = float(val)
                if split_key == '<=':
                    next_subtree = tree[attr]['<= ' + split_val] if numeric_val <= float(split_val) else tree[attr]['> ' + split_val]
                elif split_key == '>':
                    next_subtree = tree[attr]['> ' + split_val] if numeric_val > float(split_val) else tree[attr]['<= ' + split_val]
                else:
                    return None
            except ValueError:
                return None
        else:
            next_subtree = tree[attr][val]

        if isinstance(next_subtree, dict):
            return self.tracing(next_subtree, row)
        else:
            return next_subtree


    def id3(self, dataset, data, attributes):
        target = data[data.columns[-1]].tolist()
        counters = Counter(target)

        if len(set(target)) == 1:
            return [target[0], counters[target[0]]]
        if len(attributes) == 0:
            return [counters.most_common(1)[0][0], len(target)]

        best = self.best_attribute(data, attributes)
        node = {best: {}}

        if data[best].dtype in ('int64', 'float64'):
            split_val = self.best_split_val(data, best)
            for operator, subset_condition in [('<=', data[best] <= split_val),('>', data[best] > split_val)]:
                subset = data[subset_condition]
                remaining_attributes = attributes.copy()
                remaining_attributes.remove(best)
                node[best][f'{operator} {split_val}'] = self.id3(dataset, subset, remaining_attributes)
        else:
            for val in dataset[best].unique():
                subset = data[data[best] == val]
                if len(subset) == 0:
                    node[best][val] = [counters.most_common(1)[0][0], 0]
                else:
                    remaining_attributes = attributes.copy()
                    remaining_attributes.remove(best)
                    node[best][val] = self.id3(dataset, subset, remaining_attributes)

        return node


    @staticmethod
    def calc_entropy(target):
        counters = Counter(target)
        total = len(target)
        entropy = 0

        for c in counters.values():
            prob = c / total
            entropy -= prob * log2(prob)

        return entropy


    def attr_entropy(self, data, attribute):
        attr = data[attribute].unique()
        entropy_attribute = 0

        for val in attr:
            subset = data[data[attribute] == val]
            subset_target = subset[data.columns[-1]].tolist()
            subset_entropy = self.calc_entropy(subset_target)
            subset_prob = len(subset_target) / len(data)
            entropy_attribute += subset_prob * subset_entropy

        return entropy_attribute


    def best_attribute(self, data, attributes):
        target_entropy = self.calc_entropy(data[data.columns[-1]].tolist())
        information_gains = []

        for attr in attributes:
            entropy_attribute = self.attr_entropy(data, attr)
            information_gain = target_entropy - entropy_attribute
            information_gains.append(information_gain)

        best_attribute_index = information_gains.index(max(information_gains))
        return attributes[best_attribute_index]



    def gain(self, parent_target, target1, target2):
        parent_entropy = self.calc_entropy(parent_target)
        
        total = len(parent_target)
        weight1 = len(target1) / total
        weight2 = len(target2) / total
        
        entropy1 = self.calc_entropy(target1)
        entropy2 = self.calc_entropy(target2)
        
        information_gain = parent_entropy - (weight1 * entropy1) - (weight2 * entropy2)
        
        return information_gain


    def __str__(self):
        return self.tree_to_string(self.tree)


    def tree_to_string(self, t=None, indent='', counter=1):
        if t is None:
            t = self.tree

        result = ""
        if not isinstance(t, dict):
            result += f"{indent}{t[0]} (counter{counter} = {t[1]})\n"
            return result

        for attribute, subtree in t.items():
            result += f"{indent}<{attribute}>\n"
            for val, subsubtree in subtree.items():
                result += f"{indent}    {val}:\n"
                if isinstance(subsubtree, dict):
                    result += self.tree_to_string(subsubtree, indent + '        ', counter)
                else:
                    result += f"{indent}        {subsubtree[0]} (counter{counter} = {subsubtree[1]})\n"
                counter += 1
        return result


    @staticmethod
    def bool_to_string(dataset):
        boolean_columns = dataset.select_dtypes(include=bool).columns

        for column in boolean_columns:
            dataset[column] = dataset[column].map({False: 'False', True: 'True'})

        return dataset


    @staticmethod
    def accuracy(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

