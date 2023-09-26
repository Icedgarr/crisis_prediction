import datetime
import re
from datetime import date

import numpy as np
import pandas as pd

from crisis_prediction.features.config import START_DATE


def dummitize(data, column_names):
    """Takes a pandas.Series (data) and column names
    and returns the dummitized version with all columns.
    """
    features = pd.DataFrame(columns=column_names)
    features = pd.concat([features, pd.get_dummies(data, prefix=data.name)], sort=False).fillna(0)
    return features[column_names]


def add_missing_columns(df, columns):
    return df.T.reindex(columns).T.fillna(0)


def convert_camel_case_column_to_snake_case(string):
    """
    Takes a string and converts it to snake case, removing everything after '/' character and changing spaces for '_'
    """
    string = string.split(' /')[0]
    regex = re.compile('[^a-zA-Z\s_]')
    string = regex.sub('', string)
    string = re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()
    string = re.sub('([a-z])([\s])([a-z])', r'\1_\3', string)
    regex = re.compile('[\s]')
    string = regex.sub('', string)
    return string


def isocalendar_week(date):
    return date.isocalendar()[1]


def isocalendar_year(date):
    return date.isocalendar()[0]


def isnan(num):
    return num != num


def to_year_month(yearmonth):
    if isnan(yearmonth):
        return yearmonth
    else:
        yearmonth = str(yearmonth)
        return date(int(yearmonth[:4]), int(yearmonth[4:6]), 1)


def is_multiindex(index):
    return isinstance(index, pd.MultiIndex)


def first_known_to_date(first_known):
    return max(START_DATE, datetime.date(int(first_known[:4]), int(first_known[-2:]), 1))


def create_patients_data(patient, first_known):
    if first_known.month >= 10:
        month = str(first_known.month)
    else:
        month = '0' + str(first_known.month)
    patients_data = pd.DataFrame({
        'anonymous_pat_id': [patient],
        'first_year_month': str(first_known.year) + month
    })
    return patients_data


def iso_to_gregorian(iso_year, iso_week, iso_day):
    jan4 = datetime.date(iso_year, 1, 4)
    start = jan4 - datetime.timedelta(days=jan4.isoweekday() - 1)
    return start + datetime.timedelta(weeks=iso_week - 1, days=iso_day - 1)


def cap_dates(dates):
    return dates.astype(str).apply(lambda x: x.replace("2999", "2200"))


def get_month(week):
    return int(np.ceil(week / 4.42))


def difference_in_months(date_start, date_end):
    return (date_end.year - date_start.year) * 12 + date_end.month - date_start.month
