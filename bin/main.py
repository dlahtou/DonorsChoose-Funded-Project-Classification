import pandas as pd
import psycopg2
import numpy as np
import logging
import nltk
import re
from collections import defaultdict
from sqlalchemy import create_engine, inspect
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import Imputer, StandardScaler, LabelBinarizer, MultiLabelBinarizer
import models
import transformers
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle as pkl

def main():
    logging.basicConfig(level=logging.DEBUG)
    
    # Cheat transform by loading transformed dataset
    # observations = pkl.load(open('observations.pkl', 'rb'))

    observations = extract()

    observations, mapper = transform(observations)
    gb_mapper = gb_transform(observations)

    observations, results = model(observations, mapper, gb_mapper)

    load(observations, mapper, gb_mapper, results)

def extract():
    logging.info('Begin extract')
    query_WA_schools = '''
        SELECT
                p.project_cost,
                p.project_grade_level_category AS grade_level,
                p.project_current_status AS funded,
                p.project_subject_category_tree AS subject_category,
                p.project_subject_subcategory_tree AS subject_subcategory,
                p.project_resource_category AS resource_category,
                p.project_title,
                p.project_essay,
                s.school_percentage_free_lunch AS perc_lunch,
                s.school_metro_type AS metro,
                r.num_items
        FROM projects AS p
        JOIN schools s ON   p.school_id = s.school_id
        JOIN teachers t ON p.teacher_id = t.teacher_id
        JOIN 
            (SELECT project_id, SUM(resource_quantity) AS num_items
                FROM resources
                GROUP BY project_id) AS r
            ON p.project_id = r.project_id
        WHERE s.school_state = 'Washington'
        '''
    
    cnx = create_engine('postgresql://user1:password@localhost/mcnulty', isolation_level="READ COMMITTED")
    conn = cnx.connect()
    observations = pd.read_sql_query(query_WA_schools, cnx)
    

    logging.info('End extract')
    return observations

def transform(observations):
    logging.info('Begin transform')
    
    # Remove projects listed as archived or live
    observations = observations[(observations['funded'] != 'Archived')
         & (observations['funded'] != 'Live')].copy()
    
    # Add word length columns for essay and title
    observations = transformers.add_word_counts(observations)

    # Add parts of speech count columns for essay and title
    observations = transformers.count_parts_of_speech(observations)

    # Add parts of speech density columns for essay and title
    observations = transformers.parts_of_speech_density(observations)

    # Reformat list columns
    observations = transformers.split_list_columns(observations)
    
    # Reformat cost column to float
    observations = transformers.reformat_cost(observations)

    # Binarize funded column
    fund_binarize = lambda row: 1 if row['funded'] == 'Fully Funded' else 0
    observations['funded'] = observations.apply(fund_binarize, axis=1)

    # Fill nulls in resource column
    observations['resource_category'] = observations['resource_category'].copy().fillna('Unknown')
    
    # Create mapper
    mapper = DataFrameMapper([
        (['project_cost'], StandardScaler()),
        (['perc_lunch'], [Imputer(strategy='median'), StandardScaler()]),
        (['grade_level'], LabelBinarizer()),
        (['metro'], LabelBinarizer()),
        ('subject_category', MultiLabelBinarizer()),
        ('subject_subcategory', MultiLabelBinarizer()),
        (['resource_category'], LabelBinarizer()),
        (['title_length'], StandardScaler()),
        (['title_noun_density'], StandardScaler()),
        (['title_qualifier_density'], StandardScaler()),
        (['title_verb_density'], StandardScaler()),
        (['essay_length'], StandardScaler()),
        (['essay_noun_density'], StandardScaler()),
        (['essay_qualifier_density'], StandardScaler()),
        (['essay_verb_density'], StandardScaler()),
        (['num_items'], [Imputer(strategy='median'), StandardScaler()])
    ])
    logging.info('End transform')
    return observations, mapper

def gb_transform(observations):
    '''Return a mapper object with unscaled features. Appropriate for Random Forest and Gradient Boosting models.
    '''
    logging.info('Begin GB transform')
    # Create mapper
    gb_mapper = DataFrameMapper([
        (['project_cost'], None),
        (['perc_lunch'], Imputer(strategy='median')),
        (['grade_level'], LabelBinarizer()),
        (['metro'], LabelBinarizer()),
        ('subject_category', MultiLabelBinarizer()),
        ('subject_subcategory', MultiLabelBinarizer()),
        (['resource_category'], LabelBinarizer()),
        (['title_length'], None),
        (['title_noun_density'], None),
        (['title_qualifier_density'], None),
        (['title_verb_density'], None),
        (['essay_length'], None),
        (['essay_noun_density'], None),
        (['essay_qualifier_density'], None),
        (['essay_verb_density'], None),
        (['num_items'], Imputer(strategy='median'))
    ])
    logging.info('End transform')

    return gb_mapper

def model(observations, mapper, gb_mapper):
    results = list()

    # Transform dataset
    X = mapper.fit_transform(observations)
    X_gb = gb_mapper.fit_transform(observations)

    # Extract response
    y = observations['funded']

    # Oversample minority class to balance dataset
    #X, y = transformers.oversample(X, y)

    # Train/Test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    X_gbtrain, X_gbtest = train_test_split(X, random_state=10)

    # Make a dictionary of models
    model_functions = {'lr_model': models.generate_lr_model,
        'gb_model': models.generate_gb_model,
        'rf_model': models.generate_rf_model,
        'svm_model': models.generate_svm_model,
        'knn_model': models.generate_knn_model
        }

    # For loop across models
    for (model_name, model_function) in model_functions.items():

        # Results dict
        local_dict = dict()
        local_dict['model_label'] = model_name

        # Create and train model
        if model_name in ['gb_model', 'rf_model']:
            estimator = model_function(X_gbtrain, y_train)
        else:
            estimator = model_function(X_train, y_train)
        local_dict['estimator'] = estimator

        # Store results
        if model_name in ['gb_model', 'rf_model']:
            test_preds = estimator.predict(X_gbtest)
        else:
            test_preds = estimator.predict(X_test)
        local_dict['precision'] = metrics.precision_score(y_test, test_preds)
        local_dict['recall'] = metrics.recall_score(y_test, test_preds)

        if model_name in ['gb_model', 'rf_model']:
            local_dict['auc'] = metrics.roc_auc_score(y_test, estimator.predict_proba(X_gbtest)[:, 1])
        else:
            local_dict['auc'] = metrics.roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
        results.append(local_dict)
        
    # Convert results into DataFrame
    results = pd.DataFrame(results)

    logging.info('End model')
    return observations, results

def load(observations, mapper, gb_mapper, results):
    with open('output/observations.pkl', 'wb') as open_file:
        pkl.dump(observations, open_file)
    with open('output/mapper.pkl', 'wb') as open_file:
        pkl.dump(mapper, open_file)
    with open('output/gb_mapper.pkl', 'wb') as open_file:
        pkl.dump(gb_mapper, open_file)
    with open('output/results.pkl', 'wb') as open_file:
        pkl.dump(results, open_file)
    pass

if __name__ == '__main__':
    main()