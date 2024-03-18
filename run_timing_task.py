import argparse
import subprocess


def get_parser_conf_set(parser):
    parser_conf_set = parser.add_parser(
        "conf_set",
        help="Gets the conflict set of a bank given an address in a bank. The output file contains addresses offsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_conf_set.add_argument(
        "--range",
        type=int,
        help="The amount of bytes to iterate over.",
        default=8388608,
    )
    parser_conf_set.add_argument(
        "--it",
        type=int,
        help="Number of iterations when confirming conflict timing",
        default=10,
    )
    parser_conf_set.add_argument(
        "--step",
        type=int,
        help="How many step bytes to step over for each iteration.",
        default=32,
    )
    parser_conf_set.add_argument(
        "--threshold",
        type=int,
        help="Time value to be considered a conflict.",
        default=640,
    )
    parser_conf_set.add_argument(
        "--trgtBankOfs",
        type=int,
        help="Byte offset for address of the target bank. The program will get the offsets in the same bank as this address.",
        default=0,
    )
    parser_conf_set.add_argument(
        "--file",
        type=str,
        help="File to store offset.",
        default="CONF_SET.txt",
    )


def get_parser_row_set(parser):
    parser_row_set = parser.add_parser(
        "row_set",
        help="Finds the rows in the bank given conflict set. Outputs file with row addresses and offset between previous row address found",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_row_set.add_argument(
        "--it",
        type=int,
        help="Number of iterations when confirming conflict timing",
        default=10,
    )
    parser_row_set.add_argument(
        "--threshold",
        type=int,
        help="Time value to be considered a conflict.",
        default=640,
    )
    parser_row_set.add_argument(
        "--trgtBankOfs",
        type=int,
        help="Byte offset for address of the target bank. The program will get the offsets in the same bank as this address.",
        default=0,
    )
    parser_row_set.add_argument(
        "--max",
        type=int,
        help="Maximum number of rows we will get.",
        default=0,
    )
    parser_row_set.add_argument(
        "inputFile",
        type=str,
        help="File to store offset.",
    )
    parser_row_set.add_argument(
        "--outputFile",
        type=str,
        help="File to store offset.",
        default="ROW_SET.txt",
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
        prog="run_timing_task",
        description="Facade Runner that executes different CUDA timing channel tasks",
    )
    subparsers = parser.add_subparsers(
        dest="task_name", help="use `<command> -h` to see respective arguments"
    )
    get_parser_conf_set(subparsers)
    get_parser_gen_time(subparsers)
    get_parser_row_set(subparsers)

    args = parser.parse_args()
    match args.task_name:
        case "conf_set":
            p = subprocess.Popen(
                f"./out/build/rbce_conf_set {args.range} {args.it} {args.step} {args.threshold} {args.trgtBankOfs} {args.file}",
                shell=True,
            )
            p.wait()
        case "row_set":
            p = subprocess.Popen(
                f"./out/build/rbce_row_set {args.it} {args.threshold} {args.trgtBankOfs} {args.max} {args.inputFile} {args.outputFile}",
                shell=True,
            )
            p.wait()
        case "gt":
            p = subprocess.Popen(
                f"./out/build/rbce_gen_time {args.range} {args.it} {args.step} {args.file}",
                shell=True,
            )
            p.wait()
