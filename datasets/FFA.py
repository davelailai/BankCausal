
from mmpretrain.datasets.multi_label import MultiLabelDataset
# from typing import List

from mmpretrain.registry import DATASETS
from mmpretrain.datasets.base_dataset import BaseDataset

import copy
from typing import Callable, List, Union,Optional, Sequence
from mmcv.transforms import Compose

from mmcv.transforms import BaseTransform
from mmpretrain.datasets.transforms import MultiView
from mmpretrain.registry import TRANSFORMS
import numpy as np
from torch.utils.data import Dataset
import torch
# from mmengine.registry import TRANSFORMS
# from mmengine.dataset import BaseTransform
# from mmengine.registry import TRANSFORMS

# from mmcv.transforms import TRANSFORMS
# Define type of transform or transform config
Transform = Union[dict, Callable[[dict], dict]]

# class Compose:
#     """Compose multiple transforms sequentially.

#     Args:
#         transforms (Sequence[dict, callable], optional): Sequence of transform
#             object or config dict to be composed.
#     """

#     def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
#         self.transforms: List[Callable] = []

#         if transforms is None:
#             transforms = []

#         for transform in transforms:
#             # `Compose` can be built with config dict with type and
#             # corresponding arguments.
#             if isinstance(transform, dict):
#                 transform = TRANSFORMS.build(transform)
#                 if not callable(transform):
#                     raise TypeError(f'transform should be a callable object, '
#                                     f'but got {type(transform)}')
#                 self.transforms.append(transform)
#             elif callable(transform):
#                 self.transforms.append(transform)
#             else:
#                 raise TypeError(
#                     f'transform must be a callable object or dict, '
#                     f'but got {type(transform)}')

#     def __call__(self, data: dict) -> Optional[dict]:
#         """Call function to apply transforms sequentially.

#         Args:
#             data (dict): A result dict contains the data to transform.

#         Returns:
#            dict: Transformed data.
#         """
#         for t in self.transforms:
#             data = t(data)
#             # The transform will return None when it failed to load images or
#             # cannot find suitable augmentation parameters to augment the data.
#             # Here we simply return None if the transform returns None and the
#             # dataset will handle it by randomly selecting another data sample.
#             if data is None:
#                 return None
#         return data

#     def __repr__(self):
#         """Print ``self.transforms`` in sequence.

#         Returns:
#             str: Formatted string.
#         """
#         format_string = self.__class__.__name__ + '('
#         for t in self.transforms:
#             format_string += '\n'
#             format_string += f'    {t}'
#         format_string += '\n)'
#         return format_string


@DATASETS.register_module()
class MultiLabel_Ffa(MultiLabelDataset):
    
    
     def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image categories of specified index.
        """
        return self.get_data_info(idx)['gt_score']
     
     
@TRANSFORMS.register_module()
class MultiLabelMultiView(BaseTransform):
    """A transform wrapper for multiple views of an image.

    Args:
        transforms (list[dict | callable], optional): Sequence of transform
            object or config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            ``transform`` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.

    Examples:
        >>> # Example 1: MultiViews 1 pipeline with 2 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=2,
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224))],
        >>>         ])
        >>> ]
        >>> # Example 2: MultiViews 2 pipelines, the first with 2 views,
        >>> # the second with 6 views
        >>> pipeline = [
        >>>     dict(type='MultiView',
        >>>         num_views=[2, 6],
        >>>         transforms=[
        >>>             [
        >>>                dict(type='Resize', scale=224)],
        >>>             [
        >>>                dict(type='Resize', scale=224),
        >>>                dict(type='RandomSolarize')],
        >>>         ])
        >>> ]
    """

    def __init__(self, transforms: List[List[Transform]],
                 num_views: Union[int, List[int]]) -> None:

        if isinstance(num_views, int):
            num_views = [num_views]
        assert isinstance(num_views, List)
        assert len(num_views) == len(transforms)
        self.num_views = num_views

        self.pipelines = []
        for trans in transforms:
            pipeline = Compose(trans)
            self.pipelines.append(pipeline)

        self.transforms = []
        for i in range(len(num_views)):
            self.transforms.extend([self.pipelines[i]] * num_views[i])

    def transform(self, results: dict) -> dict:
        """Apply transformation to inputs.

        Args:
            results (dict): Result dict from previous pipelines.

        Returns:
            dict: Transformed results.
        """
        multi_views_outputs = dict(img=[])
        for trans in self.transforms:
            inputs = copy.deepcopy(results)
            outputs = trans(inputs)

            multi_views_outputs['img'].append(outputs['img'])
        results.update(multi_views_outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + '('
        for i, p in enumerate(self.pipelines):
            repr_str += f'\nPipeline {i + 1} with {self.num_views[i]} views:\n'
            repr_str += str(p)
        repr_str += ')'
        return repr_str