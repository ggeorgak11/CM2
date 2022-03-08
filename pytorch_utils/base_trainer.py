from __future__ import division
import sys
import time
import signal

import os
import subprocess

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from tensorboardX import SummaryWriter

from pytorch_utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer(object):
    """ BaseTrainer class to be inherited

    options
    - time_to_run
    - checkpoint_dir
    - summary_dir
    - checkpoint
    - resume
    - num_epochs
    - batch_size
    - num_workers
    - pin_memory
    - shuffle_train
    - summary_steps
    - checkpoint_steps
    - test_steps

    init_fn needs to define:
    - models_dict
    - optimizers_dict
    - train_ds - training dataset to feed to the CheckpointDataLoader
    - cdl_kwargs - Optional - kwargs to feed to the CheckpointDataLoader
                   there are defaults available and can be overwritten
                   or just added to
    """

    def __init__(self, options):
        self.options = options

        with open(os.path.join(self.options.log_dir, "git_hash.txt"), 'a') as f:
            output = subprocess.Popen(['git', 'rev-parse', 'HEAD'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
            commit_hash = output.communicate()
            commit_hash = commit_hash[0].decode('ascii')
            print("commit_hash", commit_hash)
            f.write(commit_hash)

        self.endtime = time.time() + self.options.time_to_run

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # keyword arguments for CheckpointDataLoader in the training
        # loop
        self.cdl_kwargs = {
                    "batch_size": self.options.batch_size,
                    "num_workers": self.options.num_workers,
                    "pin_memory": self.options.pin_memory,
                    "shuffle": self.options.shuffle_train
                }
        if self.options.pad_text_feat:
            if self.options.vln or self.options.vln_no_map:
                self.cdl_kwargs['collate_fn'] = PadCollate(item_keys=['tokens_tensor', 'segments_tensors'], dim=1)
            else:    
                self.cdl_kwargs['collate_fn'] = PadCollate(item_keys=['text_feat'], dim=0)

        # override this function to define your model, optimizers etc.
        self.init_fn()

        self.models_dict = {k:v.to(self.device)
                for k,v in self.models_dict.items()}

        # tensorboardX SummaryWriter for use in train_summaries
        self.summary_writer = SummaryWriter(self.options.summary_dir)

        # Load the latest checkpoints
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict,
                    self.optimizers_dict,
                    checkpoint_file=self.options.checkpoint)

        # Reload epoch and step count if a checkpoint was loaded
        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

        self.lr_schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(v,
                                                gamma=self.options.lr_decay,
                                                last_epoch=self.epoch_count-1)\
                              for k,v in self.optimizers_dict.items()}

        #for opt in self.optimizers_dict: # scheduler step has to be done after optimizer step
        #    self.lr_schedulers[opt].step()

        self.exit_code = None
        self.internal_exit = True
        signal.signal(signal.SIGTERM, self.safe_exit)

    def safe_exit(self, signal_num):
        print("Received signal", signal_num)
        self.exit_code = 0

    def train(self):
        # Create the dataloader that will generate the data
        # permutation for each epoch
        train_data_loader = CheckpointDataLoader(self.train_ds,
                **self.cdl_kwargs)
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs),
                total=self.options.num_epochs, initial=self.epoch_count):
            # setup the next epoch inside of train_data_loader
            # this will include the next dataset permutation
            train_data_loader.next_epoch(self.checkpoint)

            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime and self.exit_code is None:
                    batch = {k: v.to(self.device) for k,v in batch.items()}
                    torch.cuda.empty_cache()
                    out = self.train_step(batch, self.step_count)

                    self.step_count += 1
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch,
                                             self.step_count % self.options.image_summary_steps==0,
                                             *out)
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict,
                                self.optimizers_dict, epoch, step+1,
                                self.options.batch_size,
                                train_data_loader.get_dataset_perm(),
                                self.step_count)

                        tqdm.write('Checkpoint saved')

                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                elif self.exit_code is not None:
                    tqdm.write('Job preempted')
                    self.saver.save_checkpoint(self.models_dict,
                            self.optimizers_dict, epoch, step,
                            self.options.batch_size,
                            train_data_loader.get_dataset_perm(),
                            self.step_count)
                    tqdm.write('Checkpoint saved')
                    if self.internal_exit:
                        sys.exit(self.exit_code)
                    else:
                        return self.exit_code
                else:
                    tqdm.write('Timeout reached')
                    self.saver.save_checkpoint(self.models_dict,
                            self.optimizers_dict, epoch, step,
                            self.options.batch_size,
                            train_data_loader.get_dataset_perm(),
                            self.step_count)
                    tqdm.write('Checkpoint saved')
                    self.exit_code = 3 # requeue
                    if self.internal_exit:
                        sys.exit(self.exit_code)
                    else:
                        return self.exit_code

            # apply the learning rate scheduling policy
            for opt in self.optimizers_dict:
                self.lr_schedulers[opt].step()
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict,
                        self.optimizers_dict, epoch+1, 0,
                        self.options.batch_size, None, self.step_count)

        # Done all epochs, so exit and don't requeue
        self.exit_code = 0
        if self.internal_exit:
            sys.exit(self.exit_code)
        else:
            return self.exit_code

    def get_lr(self):
        #return next(iter(self.lr_schedulers.values())).get_lr()[0]
        return next(iter(self.lr_schedulers.values())).get_last_lr()[0]

    def init_fn(self):
        raise NotImplementedError('You need to provide an init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a train_step method')

    def train_summaries(self, input_batch, model_output):
        raise NotImplementedError('You need to provide a train_summaries method')

    def test(self):
        pass


##########################
# Code for padding the text features within a batch
##########################

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, item_keys, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
            item_keys - list of the item keys to be padded (e.g. 'text_feat')
        """
        self.dim = dim
        self.item_keys = item_keys

    def pad_collate(self, batch):
        """
        args:
            batch - list of examples

        reutrn:
            data - collated batch with padded text features
        """
        for ikey in self.item_keys:
            # find longest sequence
            max_v = 0
            for i in range(len(batch)):
                v = batch[i][ikey].shape[self.dim]
                if v > max_v:
                    max_v = v

            # Pad the vectors
            for i in range(len(batch)):
                batch[i][ikey] = pad_tensor(batch[i][ikey], max_v, self.dim)

        data = {}
        for key in batch[0].keys():
            data_key = []
            for i in range(len(batch)):
                data_key.append(batch[i][key])
            data[key] = torch.stack(data_key, dim=0)

        return data

    def __call__(self, batch):
        return self.pad_collate(batch)