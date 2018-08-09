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
    
    observations = extract()

    observations, mapper = transform(observations)

    load(observations, mapper)

    #load estimator
    with open('monday_SMOTE_best_estimator.pkl', 'rb') as open_file:
        est = pkl.load(open_file)

    X = mapper.fit_transform(observations)
    y = observations['funded']

    y_preds = est.predict(X)

    print(f"precision: {metrics.precision_score(y, y_preds)}")
    print(f"recall: {metrics.recall_score(y, y_preds)}")
    print(f"auc: {metrics.roc_auc_score(y, est.predict_proba(X)[:, 1])}")


def extract():
    logging.info('Begin extract')
    query_OR_schools = '''
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
        WHERE s.school_state = 'Oregon'
        '''
    
    cnx = create_engine('postgresql://user1:password@localhost/mcnulty', isolation_level="READ COMMITTED")
    conn = cnx.connect()
    observations = pd.read_sql_query(query_OR_schools, cnx)
    

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

    # Add parts of speech density columns
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
    return observations, mapper

def load(observations, mapper):
    with open('ORobservations.pkl', 'wb') as open_file:
        pkl.dump(observations, open_file)
    with open('ORmapper.pkl', 'wb') as open_file:
        pkl.dump(mapper, open_file)
    pass

if __name__ == '__main__':
    main()