from os import environ


def assert_in(file, files_to_check):
    if file not in files_to_check:
        raise AssertionError("{} does not exist in the list".format(str(file)))
    return True


def assert_in_env(check_list: list):
    for item in check_list:
        assert_in(item, environ.keys())
    return True
