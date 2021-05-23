#https://codereview.stackexchange.com/a/20449
from os.path import dirname, join
MAIN_DIRECTORY = dirname(dirname(__file__))
def get_resource_path(*path):
    return join(MAIN_DIRECTORY, *path)