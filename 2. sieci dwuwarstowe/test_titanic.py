import pandas as pd
from model.model import dzialaj2, init2, plot_errors, ucz2


def preprocess_data(df):
    # Handling missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)

    # Converting categorical variables to numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)

    # Selecting features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    df = df[features]
    return df


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess both training and test data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    return train_df, test_df


def test_network(W1, W2, test_df):
    results = []
    for i in range(test_df.shape[0]):
        X = test_df.iloc[i].values
        _, Y2 = dzialaj2(W1, W2, X)
        results.append(Y2[0, 0])
    return results

# Assuming weights are loaded into W1, W2
# Load the model weights (example, should load from a file or other source)
# W1, W2 = load_weights_function()


# Paths to the dataset files
train_path = '../titanic/data/train.csv'
test_path = '../titanic/data/test.csv'

# Load and preprocess data
train_df, test_df = load_data(train_path, test_path)

W1, W2 = init2(2, 2, 1)

# Test the model with test data
results = test_network(W1, W2, test_df)

# Save results to a CSV file
output = pd.DataFrame({'Survived': results})
output.to_csv('../titanic/data/predictions.csv', index=False)
print("Predictions saved to predictions.csv")
