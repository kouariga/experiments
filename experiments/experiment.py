import abc
import argparse
import datetime
import json
import os
import pdb
import sys
import shutil
import traceback

import numpy as np
from pathlib import Path
import yaml

from .multiprocessing import cartesian, run_parallel


class Experiment:

    def __init__(self, config):
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
        parser.add_argument('--gpus', type=int, nargs='+',
                            help='available gpu ids')
        parser.add_argument('--ncpus', type=int, default=1,
                            help='available number of cpus')
        parser.add_argument('--runs', type=int, default=1,
                            help='number of runs for each configuration')
        args = parser.parse_args()

        config = yaml.load(open(args.config))
        configs = args.runs*cartesian(config)

        if args.gpus:
            run_parallel(cls, configs, gpus=args.gpus)
        elif args.ncpus:
            run_parallel(cls, configs, ncpus=args.ncpus)

    def run(self):
        """Wrapper for _main to handle the top level exceptions
        """
        try:
            self._make_log_dir()
            self._save_config()
            self.main()
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
    def main(self):
        """Run experiment which is implemented by the inheritor
        """
        pass

    def _make_log_dir(self):
        """Make log directory

        root/logs/params:timestamp
        """
        tstamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        params = json.dumps(self._config['hyperparams'],
                            sort_keys=True,
                            separators=(',', ':'))
        log_dir = '{}-{}'.format(tstamp, params)
        log_dir_path = Path(self._root_dir).joinpath('logs', log_dir)
        os.makedirs(log_dir_path)
        self._log_dir = log_dir_path

    def _save_config(self):
        """Save configuration file in the log directory
        """
        if self._log_dir:
            path = Path(self._log_dir).joinpath('config.yaml')
            with path.open('w') as f:
                yaml.dump(self._config, f, default_flow_style=False)

    def _save_npfile(self, file_, arr):
        """Save numpy files such as predicted labels

        Parameters
        ----------
        file_ : base file name
        arr : array-like
        """
        if self._log_dir:
            path = Path(self._log_dir).joinpath(file_)
            np.save(path, arr)

    def _clean(self):
        """Clean up log directory when experiment was aborted by an exception
        """
        if self._log_dir:
            shutil.rmtree(self._log_dir)
