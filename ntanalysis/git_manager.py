import git


def git_version():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
