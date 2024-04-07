# coding=utf-8
__author__ = "Dimitris Karkalousos"

import argparse

from atommic.cli.launch import register_cli_subcommand


def main():
    """Run the CLI."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser = parser.add_subparsers(help="atommic commands.")
    subparser.required = True
    subparser.dest = "subcommand"

    register_cli_subcommand(subparser)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
