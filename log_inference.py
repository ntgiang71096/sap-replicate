import pickle
import data_preprocessor
import numpy as np
import feature_options
import pandas as pd


def retrieve_top_features(classifier, vectorizer):
    print("Feature names with co-efficient scores:")
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(classifier.coef_[0], feature_names))
    df = pd.DataFrame(coefs_with_fns)
    df.columns = "coefficient", "word"
    df.sort_values(by="coefficient")

    df_pos = df.tail(30)
    df_pos.style.set_caption("security related words")
    print(df_pos.to_string())

    print('-' * 32)
    df_neg = df.head(30)
    df_neg.style.set_caption("security un-related words")
    print(df_neg.to_string())


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

retrieve_top_features(model, commit_message_vectorizer)