import abc
import datetime
import glob
import os
import pdb
import pickle
import sys
import shutil
import time
import traceback

import yaml

from .dataset import Dataset


class Experiment:

    def __init__(self, config, lock=None, **kwargs):

        def set_attributes(kwargs):
            for key, val in kwargs.items():
                if isinstance(val, dict):
                    set_attributes(val)
                else:
                    setattr(self, '_' + key, val)

        self._config = config
        self._lock = lock
        self._waittime = 10.0
        set_attributes(config)

    def run(self):
        """Run experiment (wrapper)
        """
        try:
            self._main()
        except KeyboardInterrupt:
            print("SIGINT was received. Aborting experiments...")
            self._clean()
        except Exception as ex:
            message = "An exception of type {0} occurred. Arguments:\n{1!r}"
            print(message.format(type(ex).__name__, ex.args))
            _, _, tb = sys.exc_info()
            traceback.format_exc()
            pdb.post_mortem(tb)
            self._clean()

    @abc.abstractmethod
    def _main(self):
        """Run experiment (main)
        """
        pass

    @abc.abstractmethod
    def _generate_model(self):
        """Generate model
        """
        pass

    @abc.abstractmethod
    def _train(self, *args, **kwargs):
        """Train model
        """
        pass

    def _load_dataset(self, model, data_dir, n_samples):
        """Load dataset

        Returns:
            datasets (tuple of Dataset): processed datasets
        """
        dataset = model.process_dataset(n_samples, data_dir=data_dir)

        return dataset

    def _make_result_dirs(self, skip=False):
        """Makes results directory which includes ckpts and log directory

        results -- ckpts  -- params:timestamp
                 L log    -- params:timestamp
        Args:
            skip (bool): If True, already trained parameters are skipped.

        Return:
            results directory (tuple of str): (ckts dir, log dir)
        """
        result_dirs = ['ckpts', 'logs']

        params = '{}_'.format(self._config['dataset'])
        hyperparams = self._config['hyperparams']
        params += '_'.join(['{}_{}'.format(key, hyperparams[key])
                            for key in sorted(hyperparams)])
        tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

        path = os.path.join(self._results_dir, 'ckpts', params + ':*')
        if skip and (len(glob.glob(path)) > 0):
            return None

        dir_name = '{}:{}'.format(params, tstamp)
        dir_paths = [os.path.join(self._results_dir, d, dir_name)
                     for d in result_dirs]
        for dir_path in dir_paths:
            os.makedirs(dir_path)

        return tuple(dir_paths)

    def _split_datasets(self, dataset, recipe):
        """Split datasets

        Args:
            dataset (Dataset): dataset to split
            recipe (dict): instruction for how to split
        """
        keys = list(recipe.keys())
        vals = list(recipe.values())
        datasets = dataset.split(vals)
        split_datasets = dict(zip(keys, datasets))

        with open(os.path.join(self._ckpt_dir, 'datasets.pkl'), 'wb') as f:
            pickle.dump(split_datasets, f)

        return split_datasets

    def _save_config(self):
        """Saves configuration file
        """
        with open(os.path.join(self._ckpt_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def _clean(self):
        """Clean up checkpoint and log directories
        """
        if self._ckpt_dir:
            shutil.rmtree(self._ckpt_dir)
        if self._log_dir:
            shutil.rmtree(self._log_dir)