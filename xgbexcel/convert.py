import json
import sys

class XGBtoExcel:

    def __init__(self, xgb_model):
        """
        Converts an XGBoost model to an Excel formula expression.
        Supports XGBRegressor and XGBClassifier (multi-class).
        """
        sys.setrecursionlimit(10000)

        # Base score
        self.const = xgb_model.base_score if xgb_model.base_score else 0.5

        # Detect if classifier
        self.num_class = getattr(xgb_model, "n_classes_", 1)
        self.is_classifier = self.num_class > 1

        # Parse trees
        self.list_node_dict = [
            json.loads(tree_json)
            for tree_json in xgb_model.get_booster().get_dump(dump_format='json')
        ]
        self.expression_trees = [str(Node(tree)) for tree in self.list_node_dict]

        # Build full expression
        self.expression = self._get_expr()

    def _get_expr(self):
        if not self.is_classifier:
            expression = self.expression_trees[0]
            for idx in range(1, len(self.expression_trees)):
                expression = f"({expression}+{self.expression_trees[idx]})"
            return f"({expression}+{self.const})"
        else:
            # CLASSIFIER LOGIC (multi-class)
            class_expressions = []
            for c in range(self.num_class):
                expr = self.expression_trees[c]
                for idx in range(c + self.num_class, len(self.expression_trees), self.num_class):
                    expr = f"({expr}+{self.expression_trees[idx]})"
                class_expressions.append(f"({expr}+{self.const})")
    
            # Softmax in Excel
            denom = "+".join([f"EXP({ce})" for ce in class_expressions])
            softmax_exprs = [f"(EXP({ce})/({denom}))" for ce in class_expressions]
            return " , ".join(softmax_exprs)  # comma-separated probabilities


    def __str__(self):
        return self.expression

    def rename_features(self, feature_names):
        """
        A method that takes an expression and a dictionary of feature names and 
        returns a new expression with the features renamed based on the dictionary
        """
        # Iterate through the features and rename thm
        for old_name, new_name in feature_names.items():
            self.expression = self.expression.replace(old_name, new_name)
        
        print('Feature are renamed!')

    def save_expr(self, outFile):
        with open(outFile, 'w') as f:
            f.write(self.expression)
        print("Saved!")


class Node:

    def __init__(self, node_dict, sep=','):
        node_keys = ["nodeid", "depth", "split", "split_condition", "yes", "no", "children"]
        self.isLeaf = "leaf" in node_dict
        self.sep = sep

        if self.isLeaf:
            self.nodeid = node_dict["nodeid"]
            self.leaf_value = node_dict["leaf"]
        else:
            self.nodeid = node_dict["nodeid"]
            self.depth = node_dict["depth"]
            self.split = f'x{int(node_dict["split"].replace("f", "")) + 1}'
            self.split_condition = node_dict["split_condition"]
            self.left_child_id = node_dict["yes"]
            self.right_child_id = node_dict["no"]
            self.left_children_node_dict = node_dict["children"][0]
            self.right_children_node_dict = node_dict["children"][1]

    def __str__(self):
        if self.isLeaf:
            return str(self.leaf_value)
        else:
            return f'IF(({self.split}<={self.split_condition}){self.sep}{Node(self.left_children_node_dict)}{self.sep}{Node(self.right_children_node_dict)})'
