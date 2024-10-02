class Node:
    """Tree node class that stores the feature and threshold for the split, and the branches."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


def fit_decision_tree(data, features, target, depth=0, max_depth=2):
    # This function should implement the recursive process to construct a decision tree using Hunt's algorithm
    
    # Implement the following logic in the "if Conditions": 
    # If reaches a leaf node (Target labels are from a single class), or max_depth, return current label
    # In case it reaches max_depth, return majority class label

    ### leaf node : If all classes are the same, there is no need for further splitting. 
    ### set(data[target]) : To remove duplicates and extract only the unique values.
    if len(set(data[target]))==1:
        # Implement code to return the label

        ### data[target].iloc[0] : all the numbers in data[target] are the same, so you can use the first element.
        return Node(label=data[target][0])
    
    if depth>=max_depth:

        ### Version 1 : value_counts / IndexError: list index out of range
        ### data.value_counts(target).idxmax()[0] : get the most frequent value
        ### You need to convert the data into a DataFrame to use "value_counts"
        # data=pd.Series(data)
        # return Node(label=data.value_counts(target).idxmax()[0])

        ### Version 2 : mode()
        ### When the dataset is DataFrame, you can use mode()
        # return Node(label=data[target].mode()[0])

        ### Version 3 : max()
        return Node(label=max(set(data[target]), key=data[target].count))


    # Select the best split based on median
    best_feature, best_threshold = None, None
    min_gini = float('inf')
    best_left_data, best_right_data = None, None

    # Implement the major for loop to select the best feature to be used here:
    for feature in features:
        # Calculate the median of the current feature
        threshold = median(data[feature])
        values = data[feature]
        
        left_data = {'Feature1': [], 'Feature2': [], 'Target': []}
        right_data = {'Feature1': [], 'Feature2': [], 'Target': []}

        # Implement spliting the current data based on the threshold

        ### Version 1
        ### When the dataset is DataFrame, you can use below codes
        # left_data=data[data[feature]<=threshold]
        # right_data=data[data[feature]>threshold]

        ### Version 2
        for i in range(len(data[feature])):
            if data[feature][i]<=threshold:
                left_data['Feature1'].append(data['Feature1'][i])
                left_data['Feature2'].append(data['Feature2'][i])
                left_data['Target'].append(data['Target'][i])
            else:
                right_data['Feature1'].append(data['Feature1'][i])
                right_data['Feature2'].append(data['Feature2'][i])
                right_data['Target'].append(data['Target'][i])


        # Implement computing Gini impurity of the split
        left_gini=gini_impurity(left_data['Target'])
        right_gini=gini_impurity(right_data['Target'])
        weighted_gini=(len(left_data)/len(data))*left_gini+(len(right_data)/len(data))*right_gini

        # Implment checking if this is the best split so far
        if weighted_gini<min_gini:
            min_gini=weighted_gini
            best_feature, best_threshold=feature, threshold
            best_left_data, best_right_data= left_data, right_data

    # Implement recursively creating nodes
    ### depth+1 : Because it goes one level deeper from the current node.
    left_node=fit_decision_tree(best_left_data,features, target, depth+1, max_depth)
    right_node=fit_decision_tree(best_right_data,features, target, depth+1, max_depth)
    
    return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)


def predict(tree, new_data):
    # This function should predict the labels for new data points using the decision tree
    # Implement to predict the labels for new data points here, return value should be: return tree.label
    if tree.label is not None:
        return tree.label
    
    ### to move right or left
    if new_data[tree.feature]<=tree.threshold:
        return predict(tree.left, new_data)
    else:
        return predict(tree.right, new_data)
    


def gini_impurity(classes):
    """Calculate the Gini impurity for a list of classes."""
    # Implement gini impurity here
    classes_counts={}
    for label in classes:
        if label not in classes_counts:
            classes_counts[label]=0
        classes_counts[label]+=1

    impurity=1.0
    for label in classes_counts:
        probability=classes_counts[label]/len(classes)
        impurity-=probability**2
    return impurity

    


def median(values):
    """Helper function to compute the median of a list of numbers."""
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    else:
        return sorted_values[mid]


################# 
# Evaluation code

# Training data:
### convert data into DataFrame
data = {
    'Feature1': [1.0, 1.0, 2.0, 2.0, 8.0, 8.0, 9.0, 9.0],
    'Feature2': [2.0, 8.0, 2.0, 8.0, 2.0, 8.0, 2.0, 8.0],
    'Target':  [1, 0, 1, 0, 0, 1, 0, 1]
}

tree = fit_decision_tree(data, ['Feature1', 'Feature2'], 'Target')

# Use the following data to do a testing
test_cases = [
    {'Feature1': 1.5, 'Feature2': 2.5, 'Expected': 1}, 
    {'Feature1': 8.5, 'Feature2': 1.5, 'Expected': 0}, 
    {'Feature1': 10.0, 'Feature2': 9.0, 'Expected': 1}, 
    {'Feature1': 0.5, 'Feature2': 9.5, 'Expected': 0}
]

# Function to run the test cases
def run_test_cases(tree, test_cases):
    print("Running test cases...")
    for idx, case in enumerate(test_cases):
        prediction = predict(tree, case)
        print(f"Test Case {idx + 1}: Features = ({case['Feature1']}, {case['Feature2']}), " +
              f"Predicted = {prediction}, Expected = {case['Expected']}")


run_test_cases(tree, test_cases)


