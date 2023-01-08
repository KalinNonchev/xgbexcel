import json
import sys
# from openpyxl.utils.cell import get_column_letter


class XGBtoExcel:

    def __init__(self, xgb_model):
        """
        A class that converts an XGBoost model to an Excel formula expression.

        Attributes:
            const: The base score of the XGBoost model.
            list_node_dict: A list of dictionaries, where each dictionary contains information about a tree in the XGBoost model.
            expression_trees: A list of strings, where each string is an Excel formula representation of a tree in the XGBoost model.
            expression: A string representation of the entire XGBoost model in the form of an Excel formula.
        """

        # https://github.com/konstantint/SKompiler#how-it-works
        # Set the recursion limit to a large value to avoid RecursionError
        sys.setrecursionlimit(10000)
        self.const = xgb_model.base_score if xgb_model.base_score else 0.5 # default value

        self.list_node_dict = [
            json.loads(json_dump) for json_dump in xgb_model.get_booster().get_dump(
                dump_format='json')]
        self.expression_trees = [str(Node(tree))
                                 for tree in self.list_node_dict]
        self.expression = self._get_expr()

    def _get_expr(self):
        """
        Generates an Excel formula expression for the entire XGBoost model by concatenating the strings in the expression_trees list.
        """
        expression = self.expression_trees[0]

        for idx in range(1, len(self.expression_trees)):
            expression = f"({expression}+{self.expression_trees[idx]})"

        return f"({expression}+{self.const})"

    def __str__(self):
        return self.expression

    def save_expr(self, outFile):
        with open(outFile, 'w') as f:
            f.write(self.expression)
        print("Saved!")


class Node:

    def __init__(self, node_dict, sep=','):
        """
        A class representing a node in a decision tree.

        Attributes:
            nodeid: The ID of the node.
            isLeaf: A boolean indicating whether the node is a leaf node.
            leaf_value: The value of the leaf node (only applicable if isLeaf is True).
            depth: The depth of the node in the tree (only applicable if isLeaf is False).
            split: An Excel reference to the cell containing the split condition for the node (only applicable if isLeaf is False).
            split_condition: The split condition for the node (only applicable if isLeaf is False).
            left_child_id: The ID of the left child of the node (only applicable if isLeaf is False).
            right_child_id: The ID of the right child of the node (only applicable if isLeaf is False).
            left_children_node_dict: A dictionary containing information about the left child of the node (only applicable if isLeaf is False).
            right_children_node_dict: A dictionary containing information about the right child of the node (only applicable if isLeaf is False).
        """

        node_keys = ["nodeid", "depth", "split",
                     "split_condition", "yes", "no", "children"]
        self.isLeaf = not all(key in node_dict.keys() for key in node_keys)
        self.sep = sep
        if self.isLeaf:
            self.nodeid = node_dict["nodeid"]
            self.leaf_value = node_dict["leaf"]
        else:
            self.nodeid = node_dict["nodeid"]
            self.depth = node_dict["depth"]
            # self.split = f'{get_column_letter(int(node_dict["split"].replace("f", "")) + 1)}1'
            self.split = f'x{int(node_dict["split"].replace("f", "")) + 1}'
            self.split_condition = node_dict["split_condition"]
            self.left_child_id = node_dict["yes"]
            self.right_child_id = node_dict["no"]

            self.left_children_node_dict = node_dict["children"][0]
            self.right_children_node_dict = node_dict["children"][1]

            assert len(node_dict["children"]
                       ) == 2, "There are more than 2 children!"
            assert self.left_children_node_dict["nodeid"] == self.left_child_id
            assert self.right_children_node_dict["nodeid"] == self.right_child_id

    def __str__(self):
        """
        Returns a string representation of the decision tree rooted at the node in the form of an Excel formula.
        """
        if self.isLeaf:
            return str(self.leaf_value)
        else:
            return f'IF(({self.split}<={self.split_condition}){self.sep}{Node(self.left_children_node_dict)}{self.sep}{Node(self.right_children_node_dict)})'
