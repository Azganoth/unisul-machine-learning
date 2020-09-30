import re
from pathlib import Path
from typing import Any, List, Tuple

Attributes = Tuple[Tuple[str, str], ...]
Instances = List[Tuple[Any, ...]]

dataset_comment_re = re.compile(r'^\s*%.*$', re.MULTILINE)
dataset_name_re = re.compile(r'@relation\s+(.+)', re.UNICODE)
dataset_attribute_re = re.compile(r'@attribute\s+(.+?)\s+(.+)', re.UNICODE)


def load_arff(file_path: Path):
    """Reads an arff file.

    Note
    ----
    Won't work while using spaces on names.

    Parameters
    ----------
    file_path : Path
        The location of the file.

    Returns
    -------
    tuple
        The name, attributes and instances of the dataset.
    """
    with open(file_path) as file:
        content = re.sub(dataset_comment_re, '', file.read())

        if (relation_match := dataset_name_re.match(content)) is not None:
            name = relation_match.group(1)
        else:
            name = ''
        attributes: Attributes = tuple(dataset_attribute_re.findall(content))
        instances: Instances = [
            tuple(instance.split(','))
            for instance in filter(bool, content.partition('@data')[-1].splitlines())]

        return name, attributes, instances


def save_arff(file_path: Path, name: str, attributes: Attributes, instances: Instances):
    """Writes an arff file.

    Parameters
    ----------
    file_path : Path
        The location of the file.
    name : str
        The name of the dataset.
    attributes : Attributes
        The attributes of the dataset.
    instances : Instances
        The instances of the dataset.
    """
    with open(file_path, 'w') as file:
        print(f'@relation {name}',
              '',
              *(f'@attribute {attribute_name} {attribute_type}'
                for attribute_name, attribute_type in attributes),
              '',
              '@data',
              *(','.join(map(str, instance)) for instance in instances),
              sep='\n',
              file=file)
