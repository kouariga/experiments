import abc
import argparse
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
from .multiprocessing import cartesian, run_parallel


class Experiment:

    def __init__(self, config, skip=False):
        """Initialize Experiment
        """
        self._config = config
        for key, val in config.items():
            setattr(self, '_' + key, val)

    @classmethod
    def run_all(cls):
        """Parse command line arguments and run experiment with multi-processes
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('config', type=str,
                            help='configuration file (*.yaml)')
        parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                            help='available gpu ids like 0 2 3')
        parser.add_argument('--rounds', type=int, default=1,
                            help='number of experiments for each configuration')
        args = parser.parse_args()

        config = yaml.load(open(args.config))
        configs = args.rounds*cartesian(config)

        run_parallel(cls, configs, args.gpus)

    def run(self):
        """Wrapper for _main to handle the top level exceptions
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
        """Run experiment which is implemented by the inheritor
        """
        pass

    def _make_log_dir(self, skip=False):
        """Make log directory

        experiment_root/logs/params:timestamp

        Args:
            skip (bool): already tried parameters are skipped
        Return:
            log_dir_path (str): log directory, return None upon skip
        """
        params = '{}_'.format(self._config['dataset'])
        hyperparams = self._config['hyperparams']
        params += '_'.join(['{}_{}'.format(key, hyperparams[key])
                            for key in sorted(hyperparams)])
        tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        logs_dir = os.path.join(self._root_dir, 'logs')

        # Skip if params are already experimented
        params_re = os.path.join(logs_dir, params + ':*')
        if skip and (len(glob.glob(params_re)) > 0):
            return None

        log_dir_name = '{}:{}'.format(params, tstamp)
        log_dir_path = os.path.join(logs_dir, log_dir_name)
        os.makedirs(log_dir_path)

        return log_dir_path

    def _save_config(self):
        """Save configuration file in the log directory
        """
        with open(os.path.join(self._log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def _clean(self):
        """Clean up log directory when experiment was aborted by an exception
        """
        if self._log_dir:
            shutil.rmtree(self._log_dir)
