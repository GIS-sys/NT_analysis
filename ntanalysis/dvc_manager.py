from dvc.repo import Repo


def dvc_save():
    repo = Repo(".")
    repo.add(["data", ".logs"])
    repo.commit()
    repo.push(targets=["data", ".logs"])


def dvc_load():
    repo = Repo(".")
    repo.pull(force=True, allow_missing=True)
