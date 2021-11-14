

class Configuration:
    def __init__(self, save_path='./data/pkl/feature_engineering'):
        self.save_path = save_path

    @property
    def nocturia(self):
        return {
            'time_range': ('22:00', '06:00'),
        }

    @property
    def bathroom_night(self):
        return self._template

    @property
    def bathroom_daytime(self):
        return self._template

    @property
    def body_temperature(self):
        return self._template

    @property
    def _template(self):
        return {
            'save_path': self.save_path,
            'save_name': None,
            'verbose': True,
            'refresh': False,
        }


feature_config = Configuration()