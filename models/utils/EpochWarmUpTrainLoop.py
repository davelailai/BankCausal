
import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from mmengine.runner.utils import calc_dynamic_intervals
@LOOPS.register_module()
class MultiLabelEpochWarmUpTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """
   
   
    def run_iter(self, idx, data_batch: Sequence[dict], **kwargs) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        # kwargs=[]
        # print('i am using')
        kwargs['current_epoch'] = self.epoch
        kwargs['total_epoch'] = self.max_epochs
        kwargs['total_iter'] = self.max_iters
        kwargs['current_iter'] = self.iter
        # kwargs['warmup_epoch']
        if hasattr(self.runner.model, 'module'):
            # print('i am using')
            self.runner.model.module.head.current_epoch=self.epoch
            self.runner.model.module.head.warmup_epoch=self.runner.model.module.warmup_epoch
        else:
            self.runner.model.head.current_epoch=self.epoch
            self.runner.model.head.warmup_epoch=self.runner.model.warmup_epoch
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper, **kwargs)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1


@LOOPS.register_module()
class MultiLabelWarmUpTestLoop(TestLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    # def __init__(self,
    #              runner,
    #              dataloader: Union[DataLoader, Dict],
    #              evaluator: Union[Evaluator, Dict, List],
    #              fp16: bool = False):
    #     super().__init__(runner, dataloader)

    #     if isinstance(evaluator, dict) or isinstance(evaluator, list):
    #         self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
    #     else:
    #         self.evaluator = evaluator  # type: ignore
    #     if hasattr(self.dataloader.dataset, 'metainfo'):
    #         self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
    #         self.runner.visualizer.dataset_meta = \
    #             self.dataloader.dataset.metainfo
    #     else:
    #         print_log(
    #             f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
    #             'metainfo. ``dataset_meta`` in evaluator, metric and '
    #             'visualizer will be None.',
    #             logger='current',
    #             level=logging.WARNING)
    #     self.fp16 = fp16

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict],**kwargs) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        if hasattr(self.runner.model, 'module'):
            if hasattr(self.runner.model.module, 'module'):
                self.runner.model.module.module.head.current_epoch = self.runner.model.module.module.warmup_epoch + 100
                self.runner.model.module.module.head.warmup_epoch = self.runner.model.module.module.warmup_epoch
            else:
                self.runner.model.module.head.current_epoch=self.runner.model.module.warmup_epoch+100
                self.runner.model.module.head.warmup_epoch=self.runner.model.module.warmup_epoch
        else:
            self.runner.model.head.current_epoch = self.runner.model.warmup_epoch + 100
            self.runner.model.head.warmup_epoch = self.runner.model.warmup_epoch

        # if hasattr(self.runner.model, 'module'):
        #     # print('i am using')
        #     self.runner.model.module.head.current_epoch=self.runner.model.module.warmup_epoch+10
        #     self.runner.model.module.head.warmup_epoch=self.runner.model.module.warmup_epoch
        # else:
        #     self.runner.model.head.current_epoch=self.runner.model.warmup_epoch+10
        #     self.runner.model.head.warmup_epoch=self.runner.model.warmup_epoch
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

@LOOPS.register_module()
class MultiLabelWarmUpValLoop(ValLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

   
    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        # self.runner.logger.info(f"Current Epoch: {self.runner.model.module.module.head.current_epoch}") 
        if hasattr(self.runner.model, 'module'):
            if hasattr(self.runner.model.module, 'module'):
                # self.runner.logger.info(f"Current Epoch: {self.runner.model.module.module.head.current_epoch}") 
                self.runner.model.module.module.head.current_epoch = self.runner.epoch
                self.runner.model.module.module.head.warmup_epoch = self.runner.model.module.module.warmup_epoch 
            else:
                # self.runner.logger.info(f"Current Epoch: {self.runner.epoch}") 
                self.runner.model.module.head.current_epoch=self.runner.epoch
                self.runner.model.module.head.warmup_epoch=self.runner.model.module.warmup_epoch
        else:
            self.runner.model.head.current_epoch = self.runner.epoch
            self.runner.model.head.warmup_epoch = self.runner.model.warmup_epoch
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)





