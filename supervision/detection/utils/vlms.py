from __future__ import annotations


def edit_distance(string_1: str, string_2: str, case_sensitive: bool = True) -> int:
    """
    Calculates the minimum number of single-character edits required
    to transform one string into another. Allowed operations are insertion,
    deletion, and substitution.

    Args:
        string_1 (str): The source string to be transformed.
        string_2 (str): The target string to transform into.
        case_sensitive (bool, optional): Whether comparison should be case-sensitive.
            Defaults to True.

    Returns:
        int: The minimum number of edits required to convert `string_1`
        into `string_2`.

    Examples:
        ```python
        import supervision as sv

        sv.edit_distance("hello", "hello")
        # 0

        sv.edit_distance("Test", "test", case_sensitive=True)
        # 1

        sv.edit_distance("abc", "xyz")
        # 3

        sv.edit_distance("hello", "")
        # 5

        sv.edit_distance("", "")
        # 0

        sv.edit_distance("hello world", "helloworld")
        # 1
        ```
    """
    if not case_sensitive:
        string_1 = string_1.lower()
        string_2 = string_2.lower()

    if len(string_1) < len(string_2):
        string_1, string_2 = string_2, string_1

    prev_row = list(range(len(string_2) + 1))
    curr_row = [0] * (len(string_2) + 1)

    for i in range(1, len(string_1) + 1):
        curr_row[0] = i
        for j in range(1, len(string_2) + 1):
            if string_1[i - 1] == string_2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + substitution_cost
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len(string_2)]


def fuzzy_match_index(
    candidates: list[str],
    query: str,
    threshold: int,
    case_sensitive: bool = True,
) -> int | None:
    """
    Searches for the first string in `candidates` whose edit distance
    to `query` is less than or equal to `threshold`.

    Args:
        candidates (list[str]): List of strings to search.
        query (str): String to compare against the candidates.
        threshold (int): Maximum allowed edit distance for a match.
        case_sensitive (bool, optional): Whether matching should be case-sensitive.

    Returns:
        Optional[int]: Index of the first matching string in candidates,
            or None if no match is found.

    Examples:
        ```python
        fuzzy_match_index(["cat", "dog", "rat"], "dat", threshold=1)
        # 0

        fuzzy_match_index(["alpha", "beta", "gamma"], "bata", threshold=1)
        # 1

        fuzzy_match_index(["one", "two", "three"], "ten", threshold=2)
        # None
        ```
    """
    for idx, candidate in enumerate(candidates):
        if edit_distance(candidate, query, case_sensitive=case_sensitive) <= threshold:
            return idx
    return None
