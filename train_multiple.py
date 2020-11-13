import argparse
import train

train_commands = [
    # anoCAE
    "-d LEGO_cb -a anoCAE -b 8 -l mssim -c rgb -r custom --inspect",
    # baselineCAE
    "-d LEGO_cb -a baselineCAE -b 8 -l mssim -c rgb -r custom --inspect",
    # inceptionCAE
    "-d LEGO_cb -a inceptionCAE -b 8 -l mssim -c rgb -r custom --inspect",
    # mvtecCAE
    "-d LEGO_cb -a mvtecCAE -b 8 -l mssim -c rgb -r custom --inspect",
    # resnetCAE
    "-d LEGO_cb -a resnetCAE -b 8 -l mssim -c rgb -r custom --inspect",
    # skipCAE
    "-d LEGO_cb -a skipCAE -b 8 -l mssim -c rgb -r custom --inspect",
    "-d LEGO_cb -a skipCAE -b 8 -l mssim -c rgb -e 20 -r custom --inspect",
    "-d LEGO_cb -a skipCAE -b 8 -l mssim -c rgb -e 30 -r custom --inspect",
]


parser = train.create_parser()
for command in train_commands:
    args_list = command.split(" ")
    args = parser.parse_args(args_list)
    try:
        train.main(args)
    except Exception:
        pass

