import pickle
import data_preprocessor
import numpy as np
import feature_options
import pandas as pd

model = pickle.load(open('log_classifier.sav', 'rb'))
commit_message_vectorizer = pickle.load(open('log_vectorizer.sav', 'rb'))

df = pd.read_csv('ground_truth.csv')
test_data = df['Message'].values.tolist()

options = feature_options.ExperimentOption()

test_data = [data_preprocessor.process_textual_information(message, options) for message in test_data]
commit_message_features = [commit_message_vectorizer.transform([message]).toarray()[0] for message in test_data]

commit_message_features = np.array(commit_message_features)
result = model.predict(commit_message_features)

df["pred_label"] = result
df.to_csv('ground_truth_tested.csv')
print(result)