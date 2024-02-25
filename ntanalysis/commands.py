from ntanalysis.infer import infer
from ntanalysis.prepare_data import prepare
from ntanalysis.train import train


def _infer():
    infer()


def _train():
    train()


def _prepare_data():
    prepare()


if __name__ == "__main__":
    _train()
    _infer()
