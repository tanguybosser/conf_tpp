from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunConfig:
    dataset: str
    model: str
    conformalizer: str
    alpha: float
    run_id: int = 0
    cal_size: int = None
    metrics: dict = None

    @property
    def summary_dict(self):
        d = {
            'dataset': self.dataset,
            'model': self.model,
            'conformalizer': self.conformalizer,
            'alpha': self.alpha,
            'run': self.run_id,
        }
        d['cal_size'] = self.cal_size
        return d

    def get_dir_path(self, root_dir):
        return (
            Path(root_dir)
            / self.model
            / self.dataset
            / str(self.alpha)
            / self.conformalizer
            / f'cal_size={self.cal_size}'
            / f'run={self.run_id}'
        )

    def get_path(self, root_dir):
        return self.get_dir_path(root_dir).parent / f'run={self.run_id}.pkl'

    def summary_str(self, separator=', '):
        return separator.join(f'{key}: {value}' for key, value in self.summary_dict.items())

    def to_series(self):
        return pd.Series(self.__dict__)
