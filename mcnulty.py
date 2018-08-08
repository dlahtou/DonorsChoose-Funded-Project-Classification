import re
from sklearn.preprocessing import StandardScaler, LabelBinarizer, Imputer, MultiLabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

def cost_column_to_float(df):
    def make_cost_float(row):
        cost = row['project_cost']
        cost = re.sub(r'[$,]', '', cost)
        return float(cost)

    df['project_cost'] = df.apply(make_cost_float, axis=1)
    return df


def column_to_list(df, col_name):
    df[col_name] = df.copy().apply(
        lambda row: row[col_name].split(', '), axis=1)
    return df


def reformat_columns(df):
    if 'funded' in df.columns:
        df = df[(df['funded'] != 'Archived') & (df['funded'] != 'Live')]
    if 'project_cost' in df.columns:
        df = cost_column_to_float(df)
    if 'subject_category' in df.columns:
        df = column_to_list(df, 'subject_category')
    if 'subject_subcategory' in df.columns:
        df = column_to_list(df, 'subject_subcategory')

    return df


def dc_mapper(column_names):
    '''Returns a DonorsChoose DataFrameMapper with custom mapping for each column
    '''
    column_preprocessors = {'project_cost': StandardScaler(),
                            'perc_lunch': [Imputer(strategy='median'), StandardScaler()],
                            'donation_amt': StandardScaler(),
                            'is_teacher': [LabelBinarizer(), StandardScaler()],
                            'incl_opt': [LabelBinarizer(), StandardScaler()],
                            'grade_level': [LabelBinarizer(), StandardScaler()],
                            'funded': [LabelBinarizer(), StandardScaler()],
                            'metro': [LabelBinarizer(), StandardScaler()],
                            'subject_category': MultiLabelBinarizer(),
                            'subject_subcategory': MultiLabelBinarizer(),
                            'resource_category': [LabelBinarizer(), StandardScaler()],
                            'title_length': StandardScaler(),
                            'title_nouns': StandardScaler(),
                            'title_qualifiers': StandardScaler(),
                            'title_verbs': StandardScaler(),
                            'essay_length': StandardScaler(),
                            'essay_nouns': StandardScaler(),
                            'essay_qualifiers': StandardScaler(),
                            'essay_verbs': StandardScaler(),
                            }

    column_astype = {'project_cost': ['project_cost'],
                     'perc_lunch': ['perc_lunch'],
                     'donation_amt': ['donation_amt'],
                     'is_teacher': ['is_teacher'],
                     'incl_opt': ['incl_opt'],
                     'grade_level': ['grade_level'],
                     'funded': ['funded'],
                     'metro': ['metro'],
                     'subject_category': 'subject_category',
                     'subject_subcategory': 'subject_subcategory',
                     'resource_category': ['resource_category'],
                     'title_length': ['title_length'],
                     'title_nouns': ['title_nouns'],
                     'title_qualifiers': ['title_qualifiers'],
                     'title_verbs': ['title_verbs'],
                     'essay_length': ['essay_length'],
                     'essay_nouns': ['essay_nouns'],
                     'essay_qualifiers': ['essay_qualifiers'],
                     'essay_verbs': ['essay_verbs'],
                     }

    return DataFrameMapper([(column_astype[column], column_preprocessors[column]) for column in column_names])


def dc_map_normalize_split(df, column_names, target='conv', test_size=0.2, random_state=10):
    '''Returns DataFrames X_train, X_test and Series y_train, y_test in binarized, normalized format
    uses list column_names to determine columns to return in X
    '''

    X = df[column_names]
    y = df[target]
    mapper = dc_mapper(column_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state)

    Xtrain_rescaled = pd.DataFrame(mapper.fit_transform(
        X_train.copy()), columns=mapper.transformed_names_)
    Xtest_rescaled = pd.DataFrame(mapper.transform(
        X_test.copy()), columns=mapper.transformed_names_)

    return Xtrain_rescaled, Xtest_rescaled, y_train, y_test


def simple_gradboost_test(Xtrain, Xtest, ytrain, ytest):
    GBclf = GradientBoostingClassifier(
        random_state=10, subsample=0.8, min_samples_split=200)
    GBclf.fit(Xtrain, ytrain)
    y_GBpred = GBclf.predict(Xtest)

    acc = round(accuracy_score(ytest, y_GBpred), 6)
    f1 = round(f1_score(ytest, y_GBpred), 6)
    recall = round(recall_score(ytest, y_GBpred), 6)
    prec = round(precision_score(ytest, y_GBpred), 6)
    print(f'accuracy score: {acc}')
    print(f'precision score: {prec}')
    print(f'recall score: {recall}')
    print(f'f1 score: {f1}')
    print(confusion_matrix(ytest, y_GBpred))

    return GBclf
