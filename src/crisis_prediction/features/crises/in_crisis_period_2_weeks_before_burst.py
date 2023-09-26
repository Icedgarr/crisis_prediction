import datetime

from crisis_prediction.features.crises.in_crisis_period import InCrisisPeriod


class InCrisisPeriod2WeeksBeforeBurst(InCrisisPeriod):
    def __init__(self, end_date: datetime.date = datetime.date.today()):
        super().__init__(end_date=end_date, weeks_before_new_burst=2)
