import data_loader
from utils import print_line_seperator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
import numpy as np
import random
import data_preprocessor
import feature_options
import click
import utils
import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

file_path = 'MSR2019/experiment/cve_label.txt'
options = feature_options.ExperimentOption()


def preprocess_data(records, options):
    print("Start preprocessing commit messages and commit file patches...")
    records = [data_preprocessor.preprocess_single_record(record, options) for record in records]
    print("Finish preprocessing commit messages commit file patches...")
    return records


def filter_using_tf_idf_threshold(records, options):
    print("Filtering using tf-idf threshold...")
    issue_tfidf_vectorizer = TfidfVectorizer(min_df=options.min_document_frequency)
    issue_corpus = []
    record_to_corpus_id = {}

    tfidf_matrix = issue_tfidf_vectorizer.fit_transform(issue_corpus)

    feature_names = issue_tfidf_vectorizer.get_feature_names()
    for record in records:
        if record.issue_info is not None and record.issue_info != '':

            # get tf-idf score for every word in document
            doc = record_to_corpus_id[record.id]
            feature_index = tfidf_matrix[doc, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
            token_to_tfidf = {}
            for token, value in [(feature_names[i], s) for (i, s) in tfidf_scores]:
                token_to_tfidf[token] = value

            # generate new issue info contains only valuable terms
            new_issue_info = ''
            for token in record.issue_info.split(' '):
                if token in token_to_tfidf and token_to_tfidf[token] >= options.tf_idf_threshold:
                    new_issue_info = new_issue_info + token + ' '

            record.issue_info = new_issue_info

    print("Finish filtering using tf-idf threshold...")
    return records


def calculate_vocabulary(records, train_data_indices, commit_message_vectorizer, options):
    print("Calculating vocabulary")
    commit_message_vectorizer.fit([records[index].commit_message for index in train_data_indices])


def retrieve_false_positive_negative(y_pred, y_test):
    false_positives = []
    false_negatives = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test[i] == 0:
            false_positives.append(i)
        if y_pred[i] == 0 and y_test[i] == 1:
            false_negatives.append(i)

    return false_positives, false_negatives


def svm_classify(classifier, x_train, x_test, y_train, y_test):
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    return metrics.precision_score(y_true=y_test, y_pred=y_pred), \
           metrics.recall_score(y_true=y_test, y_pred=y_pred), \
           metrics.f1_score(y_true=y_test, y_pred=y_pred), y_pred


def retrieve_label(records):
    target = [record.label for record in records]
    target = np.array(target)
    return target


def calculate_log_message_feature_vector(records, commit_message_vectorizer):
    commit_message_features = [commit_message_vectorizer.transform([record.commit_message]).toarray()[0]
                               for record in records]
    commit_message_features = np.array(commit_message_features)

    return commit_message_features, retrieve_label(records)


def log_message_classify(classifier, x_train, y_train, x_test, y_test):
    precision, recall, f1, log_message_pred \
        = svm_classify(classifier, x_train, x_test, y_train, y_test)

    return precision, recall, f1, log_message_pred


def retrieve_data(records, train_data_indices, test_data_indices):
    train_data = [records[index] for index in train_data_indices]
    test_data = [records[index] for index in test_data_indices]

    return train_data, test_data


def get_list_value_from_string(input):
    return list(map(float, input.strip('[]').split(',')))


def to_record_ids(false_positives, test_data_indices):
    record_ids = []
    for index in false_positives:
        record_ids.append(test_data_indices[index].id)

    return record_ids


@click.command()
@click.option('-s', '--size', default=-1)
@click.option('--ignore_number', default=True)
@click.option('--github_issue', default=True, type=bool)
@click.option('--jira_ticket', default=True, type=bool)
@click.option('--use_comments', default=True, type=bool)
@click.option('-w', '--positive_weights', multiple=True, default=[0.5], type=float)
@click.option('--n_gram', default=1)
@click.option('--min_df', default=1)
@click.option('--use_linked_commits_only', default=False, type=bool)
@click.option('--use_issue_classifier', default=True, type=bool)
@click.option('--fold_to_run', default=10, type=int)
@click.option('--use_stacking_ensemble', default=True, type=bool)
@click.option('--dataset', default='', type=str)
@click.option('--tf-idf-threshold', default=-1, type=float)
@click.option('--use-patch-context-lines', default=False, type=bool)
def do_experiment(size, ignore_number, github_issue, jira_ticket, use_comments, positive_weights, n_gram, min_df,
                  use_linked_commits_only, use_issue_classifier, fold_to_run, use_stacking_ensemble, dataset,
                  tf_idf_threshold, use_patch_context_lines):

    global file_path
    if dataset != '':
        file_path = 'MSR2019/experiment/' + dataset

    print("Dataset: {}".format(file_path))

    options = feature_options.read_option_from_command_line(size, 0, ignore_number,
                                                            github_issue, jira_ticket, use_comments,
                                                            positive_weights,
                                                            n_gram, min_df, use_linked_commits_only,
                                                            use_issue_classifier,
                                                            fold_to_run,
                                                            use_stacking_ensemble,
                                                            tf_idf_threshold,
                                                            use_patch_context_lines)

    commit_message_vectorizer = CountVectorizer(ngram_range=(1, options.max_n_gram),
                                                min_df=options.min_document_frequency)

    records = data_loader.parse_json(file_path)

    if options.data_set_size != -1:
        records = records[:options.data_set_size]

    records = preprocess_data(records, options)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=109)

    log_classifier = svm.SVC(kernel='linear', probability=True)


    fold_count = 0

    for train_data_indices, test_data_indices in k_fold.split(records):
        fold_count += 1
        if fold_count > options.fold_to_run:
            break
        print("Processing fold number: {}".format(fold_count))
        calculate_vocabulary(records, train_data_indices, commit_message_vectorizer, options)

        train_data, test_data = retrieve_data(records, train_data_indices, test_data_indices)

        print("Calculating feature vectors")
        log_x_train, log_y_train = calculate_log_message_feature_vector(train_data, commit_message_vectorizer)
        log_x_test, log_y_test = calculate_log_message_feature_vector(test_data, commit_message_vectorizer)


        # calculate precision, recall for log message classification

        print("Training model")
        precision, recall, f1, log_message_prediction\
            = log_message_classify(log_classifier, log_x_train, log_y_train, log_x_test, log_y_test)


        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))

    print("Saving model")
    pickle.dump(commit_message_vectorizer, open('log_vectorizer.sav', 'wb'))
    pickle.dump(log_classifier, open('log_classifier.sav', 'wb'))


if __name__ == '__main__':
    do_experiment()
