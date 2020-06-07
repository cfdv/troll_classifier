#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use('ggplot')

# import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.inspection import permutation_importance, plot_partial_dependence

from imblearn.ensemble import BalancedRandomForestClassifier

import pickle

# from scipy import interp

# import string

from app import conn_mongo, MONGO_COMMENT, SAVED_MODELS
import nlp

def load_comments_to_df(query={}, fields={'_id':0}):
    comments_collection = conn_mongo(coll=MONGO_COMMENT)
    docs = list(comments_collection.find(query, fields))    
    return pd.DataFrame(docs)

def feat_eng(comms):
    # assign the label, 'troll?'' to each comment
    troll_comment_ids_set = set(comms[~comms['author'].isna()]['id'])
    comms['troll?'] = [int(mybool) for mybool in [
                        commid in troll_comment_ids_set for commid in comms['id']
                        ]]
    # is this comment in reply to a troll?
    # TODO: expand the notion of a reply to troll by including top-level comments directly 
    #       responding to posts by trolls
    comms['child_of_troll?'] = [int(mybool) for mybool in [
                        pid.split('_')[1] in troll_comment_ids_set for pid in comms['parent_id']
                        ]]
    # did a troll reply to this?
    # TODO: can we identify troll comments by classifying the comments that attract trolls?
    troll_parent_ids_set = set([
                        p.split('_')[1] for p in comms[~comms['author'].isna()]['parent_id']]
                            )
    comms['parent_of_troll?'] = [int(myid in troll_parent_ids_set) for myid in comms['id']]
    # merge depth between results from psaw ('nest_level') and praw ('depth')
    comms['norm_depth'] = np.where(~comms['depth'].isna(), comms['depth'], comms['nest_level']-1)

def accuracy(y_true, y_predict):
    (TP, FP), (FN, TN) = standard_confusion_matrix(y_true, y_predict)
    num_correct = TP + TN
    num_incorrect = FP + FN
    return num_correct / (num_correct + num_incorrect)

def standard_confusion_matrix(y_true, y_predict):
    """
    y_true = [1, 1, 1, 1, 1, 0, 0]

    y_predict = [1, 1, 1, 1, 0, 0, 0]

    In [1]: standard_confusion_matrix(y_true, y_predict)
    >> array([[4., 1.],
    >>       [0., 2.]])
    """
    cm = np.zeros((2,2))
    X = np.array([y_true, y_predict])
    values, counts = np.unique(X, axis=1, return_counts=True)
    for i, v in enumerate(values.T):
        cm[tuple([1, 1] - v)] = counts[i]
    return cm.T.astype(int)

# from the lecture
# Just handy function to make our confusion matrix pretty 
def plot_confusion_matrix(cm, # confusion matrix
                          classes_x, # test to describe what the output of the classes may be (commonly 1 or 0)
                          classes_y,
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes_x))
    plt.xticks(tick_marks, classes_x, rotation=45)
    plt.yticks(tick_marks, classes_y)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,  format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')


# In[31]:


def plot_roc_nofit(ax, X_test, y_test, clf, clf_name, **kwargs):
    y_prob = np.zeros((len(y_test),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    # Predict probabilities, not classes
    y_prob = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    if len(ax.lines) == 0:
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    ax.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' % (clf_name, roc_auc))
    mean_tpr /= 1
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
#     plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", )

def evaluate_model(brf, label, X_test, y_test):
    # make predictions
    y_predict = brf.predict(X_test)
    # provide accuracy metrics
    print("\n R^2 score:", brf.score(X_test, y_test))
    print(f'\n Out of bag score: {brf.oob_score_}')
    (TP, FP), (FN, TN) = standard_confusion_matrix(y_test, y_predict)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print("  accuracy:", accuracy)
    print(" precision:", precision_score(y_test, y_predict))
    print("    recall:", recall_score(y_test, y_predict))
    # plot ROC
    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_roc_nofit(ax, X_test, y_test, brf, label)
    plt.savefig(f'img/{label}_roc.png')

def train_brf_cv(corpus, comms):
    '''
    k-folds cross-validation for model
    
    2-stage model-building process:
      1) target = 'child_of_troll?' (aka COT)
        - split/transform/fit
        - store cot_brf
      2) target = 'troll?'
        - use cot_brf to predict COT labels for corpus
        - add to comments:
          * count of replies
          * count of replies labeled as COT
        - split/transform/fit
    '''
    # pipeline parameters
    random_state = 30
    # train_test_split
    tts_param = {'test_size': 0.2, 
                 'random_state': random_state, 
                 'shuffle': True
                }
    tfidf_param = {'max_df': 0.95, 
                   'min_df': 2, 
                   'max_features': 5000, 
                   'stop_words': 'english'
                  }
    model_param = {'n_estimators': 400,
                   'max_depth': 5,
                   'max_features': 25,
                   'oob_score': True,
                   'n_jobs': -1,
                   'random_state': random_state,
                   'class_weight': 'balanced_subsample'
                  }
    comms_meta = ['controversiality', 'score', 'norm_depth']
    
    # keep a dictionary of classifiers as we go
    clfs = {}
    # train/test split
    X_train_corp, X_test_corp, y_train, y_test = train_test_split(
        corpus, comms['child_of_troll?'], **tts_param)

    #################### STAGE 1, train to classify `child_of_troll?`
    # label classifier
    clf_label = 'cot_brf'
    # tfidf vectorize and transform corpus for input to classifier
    tfidf_vectorizer = TfidfVectorizer(**tfidf_param)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_corp)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_corp)
    X_train_raw = X_train_tfidf.toarray()
    X_test_raw = X_test_tfidf.toarray()

    # add extra features to the tfidf matrix
    X_train = np.hstack([X_train_raw, comms[comms_meta].fillna(0).values[y_train.index]])
    X_test = np.hstack([X_test_raw, comms[comms_meta].fillna(0).values[y_test.index]])

    # fit the model
    brf = BalancedRandomForestClassifier(**model_param)
    brf.fit(X_train, y_train)
    clfs[clf_label] = brf
    evaluate_model(brf, clf_label, X_test, y_test)
    
    #################### STAGE 2, train to classify `troll?`
    # next engineer the `child_suggests_troll?` and `num_replies` features
    
    # TODO: make the following few commands a pre-processing function
    #   - it would vectorize, convert to array, and add metadata
    # treat the entire corpus as new data
    X_corp = corpus
    X_tfidf = tfidf_vectorizer.transform(X_corp)
    X_raw = X_tfidf.toarray()
    X = np.hstack([X_raw, comms[comms_meta].fillna(0).values])
    # predict child of troll comments
    comms['child_of_troll_pred'] = clfs['cot_brf'].predict(X)

    # TODO: there's got to be an easier way to do this...
    
    # get the num_replies to each parent_id
    count_grouped_parent_id_comms = comms.groupby('parent_id')['body'].count()
    # store the num replies in a dict
    parent_reply_counts = dict(zip(count_grouped_parent_id_comms.index, count_grouped_parent_id_comms.tolist()))
    # why? - use dict.items() and list comprehension to parse the parent_id
    #      - also preserves the parent_id
    # numpy array with top_level code (t1 or t3), parent_id, and count
    np_parent_reply_counts = np.array([k.split('_') + [v] for k, v in parent_reply_counts.items()])
    
    # TODO: see if this feature matters...for now we don't use it
    # one-hot encode 'top_level?''
    # num_parents_with_replies = len(np_parent_reply_counts)
    # top_level = np.where(np_parent_reply_counts[:,0]=='t3', 
    #                         np.ones(num_parents_with_replies), 
    #                         np.zeros(num_parents_with_replies))

    # Count the number of child_of_troll predictions for each parent_id
    parent_pred_num_child_of_troll = comms.groupby('parent_id')['child_of_troll_pred'].sum()
    parent_pred_num_child_of_troll_df = pd.DataFrame(parent_pred_num_child_of_troll)

    # join the two new features on parent_id
    # uses the parent_id preserved in the dict
    parent_id_num_replies_df = pd.DataFrame(zip(parent_reply_counts.keys(), 
                                                parent_reply_counts.values()), 
                                            columns=['parent_id', 'num_replies'])
    parent_id_num_replies_num_child_of_troll_pred = parent_id_num_replies_df.merge(
                        parent_pred_num_child_of_troll_df.reset_index(), on='parent_id')

    # parse out the short version of the parent id b/c we need to match to comment id, which has no prefix, 't1_'
    columns = list(parent_id_num_replies_num_child_of_troll_pred.columns) + ['id']
    comment_id_reply_count_troll_pred = pd.DataFrame(np.hstack(
                    [
                        parent_id_num_replies_num_child_of_troll_pred.values,
                        np.array(
                            [s[3:] for s in parent_id_num_replies_num_child_of_troll_pred['parent_id']]
                        ).reshape(-1, 1)
                    ]),
                 columns=['parent_id', 'num_replies', 'num_pred_child_of_troll', 'id']).drop('parent_id', axis=1)

    # This ensures our new features are in the original order of the corpus
    corpus_ordered_num_replies_num_pred_child_of_troll = (comms.merge(
        comment_id_reply_count_troll_pred, on='id', how='left', copy=False)
        [['num_replies', 'num_pred_child_of_troll']])

    # add to X_train
    # TODO manage the memory usage better ...
    X_train_stage2 = np.hstack([
        X_train,
        corpus_ordered_num_replies_num_pred_child_of_troll.fillna(0).values[y_train.index]
    ])
    X_test_stage2 = np.hstack([
        X_test,
        corpus_ordered_num_replies_num_pred_child_of_troll.fillna(0).values[y_test.index]
    ])

    # label classifier
    clf_label = 't_brf'
    # update the target from 'child_of_troll?' to 'troll?'
    y_train = comms['troll?'][y_train.index]
    y_test = comms['troll?'][y_test.index]
    brf = BalancedRandomForestClassifier(**model_param)
    brf.fit(X_train_stage2, y_train)
    # evaluate model
    clfs[clf_label] = brf
    evaluate_model(brf, clf_label, X_test_stage2, y_test)
    
    return clfs

def report_to_user(text):
    print(f'[run_pipeline.py] ' + text)

if __name__ == '__main__':

    report_to_user('loading comments from mongo...')
    comms = load_comments_to_df()
    # perform nlp pre-processing (lower case, remove punctuation, lemmatize, etc.)
    report_to_user('NLP pre-processing...')
    # use multiprocessing
    # TODO accept argument for number of cores to use
    agents = 30 # number of cores
    chunksize = 3500
    p = Pool(agents)
    corpus = p.map(nlp.pre_proc_doc, comms['body'], chunksize=chunksize)
    p.close()
    p.join()
    
    # add the class labels and normalized depths to the comments
    report_to_user('feature engineering...')
    feat_eng(comms)

    report_to_user('training and evaluating model...')
    clfs = train_brf_cv(corpus, comms)
    report_to_user('pickling models...')
    for label, clf in clfs.items():
        filename = SAVED_MODELS + label + '.pkl'
        with open(filename, 'wb') as fp:
            pickle.dump(clf, fp)

                # Run this with a pool of 5 agents having a chunksize of 3 until finished
#    agents = 1
#    chunksize = 100
#    p = Pool(agents)
#    p.map(job, dataset, chunksize=chunksize)
#    p.close()
#    p.join()