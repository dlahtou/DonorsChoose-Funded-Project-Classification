import re
from collections import defaultdict
import pandas as pd
import nltk
from imblearn.over_sampling import SMOTE

def add_word_counts(df):
    def word_count(text):
        return len(text.split(' '))
    df['title_length'] = df.apply(lambda row: word_count(row['project_title']), axis=1)
    df['essay_length'] = df.apply(lambda row: word_count(row['project_essay']), axis=1)

    return df

def count_parts_of_speech(df):
    '''
    Returns a dataframe with additional parts of speech columns:
    title|essay:
        verbs
        nouns
        qualifiers
    '''    

    def parts_of_speech_tuples(row, column_name):
        counts = defaultdict(int)
        text = row[column_name].lower()
        tokenized_title = nltk.word_tokenize(text)
        for word, tag in nltk.pos_tag(tokenized_title, tagset='universal'):
            counts[tag] += 1
        counts['NOUN'] += 0
        counts['QUALIFIERS'] = counts['ADV'] + counts['ADJ']
        counts['VERB'] += 0
        return (counts['NOUN'], counts['QUALIFIERS'], counts['VERB'])
    title_speech_parts = df.apply(lambda row: parts_of_speech_tuples(row, 'project_title'), axis=1)
    df['title_nouns'] = [x[0] for x in title_speech_parts]
    df['title_qualifiers'] = [x[1] for x in title_speech_parts]
    df['title_verbs'] = [x[2] for x in title_speech_parts]
    essay_speech_parts = df.apply(lambda row: parts_of_speech_tuples(row, 'project_essay'), axis=1)
    df['essay_nouns'] = [x[0] for x in essay_speech_parts]
    df['essay_qualifiers'] = [x[1] for x in essay_speech_parts]
    df['essay_verbs'] = [x[2] for x in essay_speech_parts]

    return df

def parts_of_speech_density(df):
    df['title_noun_density'] = df.apply(lambda row: row['title_nouns']/row['title_length'], axis=1)
    df['title_qualifier_density'] = df.apply(lambda row: row['title_qualifiers']/row['title_length'], axis=1)
    df['title_verb_density'] = df.apply(lambda row: row['title_verbs']/row['title_length'], axis=1)
    
    df['essay_noun_density'] = df.apply(lambda row: row['essay_nouns']/row['essay_length'], axis=1)
    df['essay_qualifier_density'] = df.apply(lambda row: row['essay_qualifiers']/row['essay_length'], axis=1)
    df['essay_verb_density'] = df.apply(lambda row: row['essay_verbs']/row['essay_length'], axis=1)

    return df

def split_list_columns(df):
    def column_to_list(df, col_name):
        df[col_name] = df.copy().apply(
            lambda row: row[col_name].split(', '), axis=1)
        return df
    df = column_to_list(df, 'subject_category')
    df = column_to_list(df, 'subject_subcategory')

    return df

def reformat_cost(df):
    def cost_to_float(cost_string):
        cost_string = re.sub(r'[$,]', '', cost_string)
        return float(cost_string)
    df['project_cost'] = df.apply(lambda row: cost_to_float(row['project_cost']), axis=1)

    return df


def oversample(X, y):
    sm = SMOTE(ratio='minority', random_state=10, kind='borderline1')
    X_res, y_res = sm.fit_sample(X, y)

    return X_res, y_res
