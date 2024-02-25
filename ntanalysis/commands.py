from ntanalysis.dvc_manager import dvc_load, dvc_save
from ntanalysis.infer import infer
from ntanalysis.train import train


def _infer():
    infer()

def _train():
    train()

if __name__ == "__main__":
    _train()
    _infer()
