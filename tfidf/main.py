import sys
import csv
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Concatenate, Input, Dropout, Dense, Dot
from keras.models import Model

BATCH_SIZE = 200
HIDDEN_DIM = 100
N_EPOCHS = 10
TEST_SPLIT_SIZE = 0.2
FEATURE_LIMIT = 5000

stance_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
stance_map_inv = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}

# Reads _bodies.csv file and creates a dictionary from Body ID -> Body Text
def get_body_dict(data_dir):
    with open(data_dir, encoding='utf_8') as tb:
        train_bodies = list(csv.reader(tb))
        train_bodies_dict = {}
        for i, line in enumerate(tqdm(train_bodies)):
            if i > 0:
                id = int(line[0])
                train_bodies_dict[id] = line[1]

    return train_bodies_dict

# Reads _stances.csv file and returns headline, body, stance data
def get_article_data(data_dir, train_bodies_dict):
    with open(data_dir, encoding='utf_8') as ts:
        train_stances = list(csv.reader(ts))

        headlines, bodies, stances = [], [], []

        for i, line in enumerate(tqdm(train_stances)):
            if i > 0:
                body_id = int(line[1].strip())

                stances.append(line[2].strip())
                headlines.append(line[0].strip())
                bodies.append(train_bodies_dict[body_id])
        return stances, headlines, bodies


def main(data_dir):
    print('Reading in CSV data...')
    train_bodies_dict = get_body_dict(os.path.join(data_dir, "train_bodies.csv"))
    train_stances, train_headlines, train_bodies = get_article_data(os.path.join(data_dir, "train_stances.csv"),
                                                                    train_bodies_dict)

    competition_bodies_dict = get_body_dict(os.path.join(data_dir, "competition_test_bodies.csv"))
    test_stances, test_headlines, test_bodies = get_article_data(os.path.join(data_dir, "competition_test_stances.csv"),
                                                                 competition_bodies_dict)

    print('Initializing TFIDF Vectorizer...')
    # Converts collection of raw documents to a TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=FEATURE_LIMIT)
    # Builds vocabulary from training set
    vectorizer.fit(train_headlines + train_bodies)

    print('Vectorizing Data...')
    # Transform documents to document-term matrix
    x_train_headlines = vectorizer.transform(train_headlines).toarray()
    x_train_bodies = vectorizer.transform(train_bodies).toarray()
    x_test_headlines = vectorizer.transform(test_headlines).toarray()
    x_test_bodies = vectorizer.transform(test_bodies).toarray()

    print('Encoding Stances...')
    # Fit encoder and return encoded labels
    encoded_train_stances = LabelEncoder().fit_transform(train_stances)
    # Transform labels to binary class matrix
    y_train = np_utils.to_categorical(encoded_train_stances, num_classes=4)
    encoded_test_stances = LabelEncoder().fit_transform(test_stances)
    y_test = np_utils.to_categorical(encoded_test_stances, num_classes=4)

    print('Creating train/test splits...')
    x_train_headlines, x_val_headlines, x_train_bodies, x_val_bodies, y_train, y_val = train_test_split(
        x_train_headlines, x_train_bodies, y_train, test_size=TEST_SPLIT_SIZE)

    print('Building Model I/O...')
    # Define model input for headlines
    input_headlines = Input(shape=(FEATURE_LIMIT,), name='input_headlines')
    # Define model input for bodies
    input_bodies = Input(shape=(FEATURE_LIMIT,), name='input_bodies')
    # Compute cosine similarity matrix
    cosine_similarity = Dot(axes=-1)([input_headlines, input_bodies])
    # Concatenate list of inputs
    concatenated_input = Concatenate()([input_headlines, input_bodies, cosine_similarity])

    # Add hidden layer
    hidden = Dense(HIDDEN_DIM, activation='sigmoid', name='dense_layer')(concatenated_input)
    # Add dropout layer
    hidden = Dropout(rate=0.6, name='dropout_layer')(hidden)
    # Add output layer
    out = Dense(4, activation='softmax', name='output_layer')(hidden)

    model = Model(inputs=[input_headlines, input_bodies], outputs=out)

    print(model.summary())

    print('Compiling Model...')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('Fitting Model...')
    model.fit([x_train_headlines, x_train_bodies], y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              validation_data=([x_val_headlines, x_val_bodies], y_val))

    print('Evaluating Model...')
    model.evaluate([x_test_headlines, x_test_bodies], y_test, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    main(sys.argv[1])
