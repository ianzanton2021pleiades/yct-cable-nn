"""Use the frozen synthetic evaluator with Iter8's 12-channel model/input."""
import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / 'FieldNN_Iter3_20261022'
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"
ORIGINAL = HERE.parent / 'FieldNN_20261022'
sys.path.extend([str(ITER5), str(ITER3), str(ORIGINAL)])
import data
import models


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--limit', type=int)
    args, rest = parser.parse_known_args(sys.argv[1:])
    if args.limit:
        base = data.ShardDataset
        class LimitedDataset(base):
            def __len__(self):
                return min(super().__len__(), args.limit)
        data.ShardDataset = LimitedDataset
    sys.argv = [str(ITER5 / 'evaluate.py'), *rest]
    runpy.run_path(str(ITER5 / 'evaluate.py'), run_name='__main__')


if __name__ == '__main__':
    main()
