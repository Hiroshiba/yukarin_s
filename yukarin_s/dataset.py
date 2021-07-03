import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy
from acoustic_feature_extractor.data.phoneme import BasePhoneme, phoneme_type_to_class
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_s.config import DatasetConfig


@dataclass
class Input:
    phoneme_list: List[BasePhoneme]


@dataclass
class LazyInput:
    phoneme_list_path: SamplingData
    phoneme_class: Type[BasePhoneme]

    def generate(self):
        return Input(
            phoneme_list=self.phoneme_class.load_julius_list(self.phoneme_list_path),
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
        phoneme_list_data: List[BasePhoneme],
        phoneme_num: int,
    ):
        assert len(phoneme_list_data) >= phoneme_num

        index = numpy.random.randint(len(phoneme_list_data) - phoneme_num + 1)
        phoneme_list_data = phoneme_list_data[index : index + phoneme_num]

        phoneme_list = numpy.array([p.phoneme_id for p in phoneme_list_data])
        phoneme_length = numpy.array([p.end - p.start for p in phoneme_list_data])

        return dict(
            phoneme_list=phoneme_list.astype(numpy.int64),
            phoneme_length=phoneme_length.astype(numpy.float32),
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
        inputs = [
            LazyInput(
                phoneme_list_path=phoneme_list_paths[fn],
                phoneme_class=phoneme_type_to_class[config.phoneme_type],
            )
            for fn in fns
        ]

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
