import argparse
import subprocess


def get_parser_gbo(parser):
    parser_gbo = parser.add_parser(
        "gbo",
        help="Gets the bank address offsets in a bank",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_gbo.add_argument(
        "--range",
        type=int,
        help="The amount of bytes to iterate over.",
        default=8388608,
    )
    parser_gbo.add_argument(
        "--it",
        type=int,
        help="Number of iterations when confirming conflict timing",
        default=10,
    )
    parser_gbo.add_argument(
        "--step",
        type=int,
        help="How many step bytes to step over for each iteration.",
        default=32,
    )
    parser_gbo.add_argument(
        "--threshold",
        type=int,
        help="Time value to be considered a conflict.",
        default=640,
    )
    parser_gbo.add_argument(
        "--trgtBankOfs",
        type=int,
        help="Byte offset for address of the target bank. The program will get the offsets in the same bank as this address.",
        default=0,
    )
    parser_gbo.add_argument(
        "--file",
        type=str,
        help="File to store offset.",
        default="SAME_BANK_ADDR.txt",
    )


def get_parser_gen_time(parser):
    parser_gt = parser.add_parser(
        "gt",
        help="Gets the timing values of all addresses on the first address. Mostly only used to get the threshold",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_gt.add_argument(
        "--range",
        type=int,
        help="The amount of bytes to iterate over.",
        default=8388608,
    )
    parser_gt.add_argument(
        "--it",
        type=int,
        help="Number of iterations when confirming conflict timing",
        default=10,
    )
    parser_gt.add_argument(
        "--step",
        type=int,
        help="How many step bytes to step over for each iteration.",
        default=32,
    )
    parser_gt.add_argument(
        "--file",
        type=str,
        help="File to store timing values",
        default="TIMING_VALUE.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="GPU Timing Channel Main",
        description="Facade Runner that executes different CUDA timing channel tasks",
    )
    subparsers = parser.add_subparsers(
        dest="task_name", help="use `<command> -h` to see respective arguments"
    )
    get_parser_gbo(subparsers)
    get_parser_gen_time(subparsers)

    args = parser.parse_args()
    match args.task_name:
        case "gbo":
            p = subprocess.Popen(
                f"./out/build/rbce_gbo {args.range} {args.it} {args.step} {args.threshold} {args.trgtBankOfs} {args.file}",
                shell=True,
            )
            p.wait()
        case "gt":
            p = subprocess.Popen(
                f"./out/build/rbce_gen_time {args.range} {args.it} {args.step} {args.file}",
                shell=True,
            )
            p.wait()
