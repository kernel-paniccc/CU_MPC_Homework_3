import argparse
from importlib import import_module
from tasks import REGISTRY

def main():
    parser = argparse.ArgumentParser(prog="worker")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    subparsers = parser.add_subparsers(dest="task", required=True)

    for name in REGISTRY.keys():
        subparsers.add_parser(name)

    dotted = subparsers.add_parser("call")
    dotted.add_argument("target", help="dotted path, напр. tasks.collective:reduce")

    args = parser.parse_args()

    if args.task == "call":
        module_path, func_name = args.target.split(":")
        fn = getattr(import_module(module_path), func_name)
    else:
        fn = REGISTRY[args.task]

    fn(rank=args.rank, world_size=args.world_size)

if __name__ == "__main__":
    main()
