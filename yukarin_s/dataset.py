import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_s.config import DatasetConfig
from yukarin_s.data.phoneme import OjtPhoneme


@dataclass
class Input:
    phoneme_list: List[OjtPhoneme]


@dataclass
class LazyInput:
    phoneme_list_path: Path

    def generate(self):
        return Input(
            phoneme_list=OjtPhoneme.load_julius_list(self.phoneme_list_path),
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        phoneme_num: int,
    ):
        self.inputs = inputs
        self.phoneme_num = phoneme_num

    @staticmethod
    def extract_input(
        phoneme_list_data: List[OjtPhoneme],
        phoneme_num: int,
    ):
        sampling_length = phoneme_num
        length = len(phoneme_list_data)

        phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list_data])
        phoneme_length = numpy.array([p.end - p.start for p in phoneme_list_data])

        if sampling_length > length:
            padding_length = sampling_length - length
            sampling_length = length
        else:
            padding_length = 0

        offset = numpy.random.randint(length - sampling_length + 1)
        offset_slice = slice(offset, offset + sampling_length)
        phoneme_list = phoneme_list[offset_slice]
        phoneme_length = phoneme_length[offset_slice]
        padded = numpy.zeros_like(phoneme_list, dtype=bool)

        pad_pre, pad_post = 0, 0
        if padding_length > 0:
            pad_pre = numpy.random.randint(padding_length + 1)
            pad_post = padding_length - pad_pre
            pad_list = [pad_pre, pad_post]
            phoneme_list = numpy.pad(phoneme_list, pad_list)
            phoneme_length = numpy.pad(phoneme_length, pad_list)
            padded = numpy.pad(padded, pad_list, constant_values=True)

        return dict(
            phoneme_list=phoneme_list.astype(numpy.int64),
            phoneme_length=phoneme_length.astype(numpy.float32),
            padded=padded,
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            phoneme_list_data=input.phoneme_list,
            phoneme_num=self.phoneme_num,
        )


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, speaker_ids: List[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.int64)
        return d


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create_dataset(config: DatasetConfig):
    phoneme_list_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_list_glob)}
    fn_list = sorted(phoneme_list_paths.keys())
    assert len(fn_list) > 0

    speaker_ids: Optional[Dict[str, int]] = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.speaker_size == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_test=False):
        inputs = [LazyInput(phoneme_list_path=phoneme_list_paths[fn]) for fn in fns]

        dataset = FeatureDataset(inputs=inputs, phoneme_num=config.phoneme_num)

        if speaker_ids is not None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        dataset = TensorWrapperDataset(dataset)

        if for_test:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_test=True),
    }
