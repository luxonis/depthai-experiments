from datetime import datetime, timedelta

CONFIDENCE_THRESHOLD = 0.5


class AlertingGate:
    def __init__(self):
        self.set_defaults()

    def set_defaults(self):
        self.last_reported = None
        self.statuses = []

    def parse_danger(self, results):
        if len(results) > 0:
            has_danger = any(map(lambda item: item["dangerous"], results))
            if has_danger:
                self.last_reported = datetime.now()

            if self.last_reported is not None:
                self.statuses = self.statuses[-50:] + [has_danger]

        if (
            self.last_reported is not None
            and datetime.now() - self.last_reported > timedelta(seconds=5)
        ):
            self.set_defaults()

        return (len(self.statuses) > 10) and (
            sum(self.statuses) / len(self.statuses) > CONFIDENCE_THRESHOLD
        )
