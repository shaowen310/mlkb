import datetime


def timedelta(timef):
    datetime.timedelta(seconds=timef)


def timedelta_str(timef):
    '{}'.format(datetime.timedelta(seconds=timef))
