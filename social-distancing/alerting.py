import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


class AlertingGate:
    confidence_threshold = 0.5

    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self.last_reported = None
        self.statuses = []

    def parse_frame(self, results):
        if len(results) > 0:
            has_danger = any(map(lambda item: item['dangerous'], results))
            if has_danger:
                self.last_reported = datetime.now()
            if self.last_reported is not None:
                self.statuses = self.statuses[-50:] + [has_danger]

        if self.last_reported is not None and datetime.now() - self.last_reported > timedelta(seconds=5):
            self.set_defaults()

        if len(self.statuses) > 10 and sum(self.statuses) / len(self.statuses) > self.confidence_threshold:
            return True
        else:
            return False


class AlertingGateDebug(AlertingGate):
    def parse_frame(self, results):
        result = super().parse_frame(results)

        last_reported_date = self.last_reported.isoformat(' ') if self.last_reported is not None else None
        positive_ratio = sum(self.statuses) / len(self.statuses) if len(self.statuses) > 0 else None
        # log.info("Result: {}, Statuses: {}, positive ratio: {}, last reported: {}".format(
        #     result, len(self.statuses), positive_ratio, last_reported_date
        # ))

        return result



