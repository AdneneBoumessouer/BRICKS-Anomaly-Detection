import argparse
import train

parser = train.create_parser()
args_list = [
    "-d",
    "LEGO_light/SV",
    "-a",
    "mvtecCAE",
    "-b",
    "8",
    "-l",
    "mssim",
    "-c",
    "rgb",
    "-e",
    100,
    "-r",
    "ktrain",
    "--inspect",
]
args = parser.parse_args(args_list)

