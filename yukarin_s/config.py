from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from yukarin_s.utility import dataclass_utility
from yukarin_s.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    root_dir: Path
    phoneme_list_pathlist_path: Path
    phoneme_num: int
    speaker_dict_path: Optional[Path]
    speaker_size: Optional[int]
    phoneme_type: str
    test_num: int
    test_trial_num: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    phoneme_size: int
    phoneme_embedding_size: int
    speaker_size: int
    speaker_embedding_size: int
    hidden_size_list: List[int]
    kernel_size_list: List[int]


@dataclass
class ModelConfig:
    eliminate_pause: bool


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    num_processes: Optional[int] = None
    use_gpu: bool = True
    use_multithread: bool = False


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "phoneme_type" not in d["dataset"]:
        d["dataset"]["phoneme_type"] = "jvs"

    if "root_dir" not in d["dataset"]:
        d["dataset"]["root_dir"] = Path(".")
    for before_key in [
        "phoneme_list_glob",
    ]:
        if before_key in d["dataset"]:
            after_key = before_key.replace("_glob", "_pathlist_path")
            d["dataset"][after_key] = d["dataset"].pop(before_key)
