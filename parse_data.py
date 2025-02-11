import csv
from sklearn.model_selection import train_test_split
from constants import TEST_DATA_SHARE, TRAIN_DATA_SHARE, FILE_NAME_INPUT_DATA


def read_csv():
    with open(FILE_NAME_INPUT_DATA, encoding='utf-8') as r_file:
        X = []
        y = []

        file_reader = csv.reader(r_file, delimiter=";")
        next(file_reader)

        for row in file_reader:
            x = [row[0], row[1], row[2], row[3]]
            X.append(x)
            y.append([row[4]])
        return X, y


def processing_data():
    X, y = read_csv()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_DATA_SHARE,
        train_size=TRAIN_DATA_SHARE,
        random_state=42
    )
    return (x_train, y_train), (x_test, y_test)