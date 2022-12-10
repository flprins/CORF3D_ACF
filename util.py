import re

RE_FROG_TANK = re.compile(r'(?P<frog>frog\d+)_(?P<tank>tank\d+)')


def filename_to_label(filename: str):
    match = RE_FROG_TANK.search(filename)
    return f'{match["frog"]}_{match["tank"]}'  # frogX_tankX
