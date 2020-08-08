import subprocess
import atexit
import pandas as pd
from datetime import datetime
from tensorboardX import SummaryWriter
from loguru import logger
import yaml
import argparse
from argparse import Namespace
import time
from tensorboardX import SummaryWriter
import os
from os.path import join
from distutils.dir_util import copy_tree

try:
    from comet_ml import Experiment as CometExperiment
    import wandb
    EXTERNAL_LOGGING_AVAILABLE = True
except Exception as e:
    EXTERNAL_LOGGING_AVAILABLE = False


def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path

    i = 1

    if os.path.isdir(fname_path):
        filename = fname_path
        new_fname = "{}-{}".format(filename, i)
    else:
        filename, file_extension = os.path.splitext(fname_path)
        new_fname = "{}-{}{}".format(filename, i, file_extension)

    while os.path.exists(new_fname):
        i += 1
        if os.path.isdir(fname_path):
            new_fname = "{}-{}".format(filename, i)
        else:
            new_fname = "{}-{}{}".format(filename, i, file_extension)

    return new_fname


class Experiment(object):
    def __init__(self, root_dir, use_comet=False, use_wandb=False):
        self.start_time = time.time()

        # define dirs
        self.dir = get_nonexistant_path(root_dir)
        self.project_name = self.dir.split('/')[-2]
        self.exp_name = self.dir.split('/')[-1]
        self.ckpt_dir = join(self.dir, 'ckpt')
        self.code_dir = join(self.dir, 'code')
        self.hparams_file = join(self.dir, 'hparams.yaml')
        self.metrics = []

        # create dirs
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.code_dir, exist_ok=True)
        copy_tree(os.path.abspath("."), self.code_dir)        
        logger.info(f"experiment folder: {self.dir}")

        # create writers
        # tensorboard
        self.tb_writer = SummaryWriter(self.dir)

        # comet_ml
        self.comet_exp = None
        if EXTERNAL_LOGGING_AVAILABLE and use_comet:
            self.comet_exp = CometExperiment(api_key="XXX",
                                             project_name=self.project_name,
                                             workspace="YYY")
            self.comet_exp.set_name(self.exp_name)
            self.comet_exp.log_parameter("exp_name", self.exp_name)

        # wandb
        self.wandb_exp = False
        if EXTERNAL_LOGGING_AVAILABLE and use_wandb:
            self.wandb_exp = True
            wandb.init(name=self.exp_name,
                       project=self.project_name,
                       dir=self.dir)

        atexit.register(self.save)

    def save_hparams(self, hparams, hparams_file=None):
        if not hparams_file:
            hparams_file = self.hparams_file

        # translate hparams into dict from various types
        if type(hparams) in [argparse.Namespace]:
            logger.info("parsing ArgumentParser hparams") 
            hparams = vars(hparams)
        elif type(hparams) == dict:
            logger.info("parsing dict hparams") 
            pass
        else:
            logger.error(f"hparams type {type(hparams)} is not supported")
            return

        # save hparams into yaml file
        with open(hparams_file, "w") as f:
            f.write(yaml.dump(hparams))
            logger.info(f"hparams file saved to: {hparams_file}")

        # log hparams into comet_ml
        if self.comet_exp:
            self.comet_exp.log_parameters(hparams)

        # log hparams into wandb
        if self.wandb_exp:
            wandb.config.update(hparams)

        logger.info("hyper-parameters:\n" + yaml.dump(hparams, default_flow_style=False)[:-1])

    @classmethod
    def load_hparams(cls, hparams_file):
        """load_hparams - returns a Namespace object
        loaded from a yaml file.

        :param hparams_file: path to yaml file
        """
        logger.info(f"loading hparams from: {hparams_file}")

        with open(hparams_file) as f:
            hparams = yaml.load(f)
            logger.info("hyper-parameters:\n" + yaml.dump(hparams, default_flow_style=False)[:-1])

            hparams = Namespace(**hparams)
            return hparams

    def log_metric(self, metrics_dict, step=None):
        # log all metrics using writers
        for k,v in metrics_dict.items():
            # log in tensorboard
            self.tb_writer.add_scalar(k, v, step)

            # log in comet_ml
            if self.comet_exp:
                self.comet_exp.log_metric(k, v, step=step)

            # log in wandb
            if self.wandb_exp:
                wandb.log({k: v}, step=step)

        self.metrics.append({**metrics_dict, **{'timestamp': str(datetime.utcnow())}})

    def save(self):
        logger.info("saving experiment")
        # save metrics to csv
        df = pd.DataFrame(self.metrics)
        df.to_csv(join(self.dir, 'metrics.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--augment', default=True, type=bool)
    args = parser.parse_args()
    
    exp = Experiment('/tmp/exp', use_comet=True, use_wandb=True)
    exp.save_hparams(args)
    exp.log_metric({'metrics/loss': 0.5})
    exp.log_metric({'metrics/loss': 0.4, 'metrics/acc': 0.99})
    exp.load_hparams(exp.dir + "/hparams.yaml")
