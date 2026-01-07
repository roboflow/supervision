import pytest

from supervision.detection.utils.vlms import edit_distance, fuzzy_match_index


@pytest.mark.parametrize(
    "string_1, string_2, case_sensitive, expected_result",
    [
        # identical strings, various cases
        ("hello", "hello", True, 0),
        ("hello", "hello", False, 0),
        # case sensitive vs insensitive
        ("Test", "test", True, 1),
        ("Test", "test", False, 0),
        ("CASE", "case", True, 4),
        ("CASE", "case", False, 0),
        # completely different
        ("abc", "xyz", True, 3),
        ("abc", "xyz", False, 3),
        # one string empty
        ("hello", "", True, 5),
        ("", "world", True, 5),
        # single character cases
        ("a", "b", True, 1),
        ("A", "a", True, 1),
        ("A", "a", False, 0),
        # whitespaces
        ("hello world", "helloworld", True, 1),
        ("test", " test", True, 1),
        # unicode and emoji
        ("😊", "😊", True, 0),
        ("😊", "😢", True, 1),
        # long string vs empty
        ("a" * 100, "", True, 100),
        ("", "b" * 100, True, 100),
        # prefix/suffix
        ("prefix", "prefixes", True, 2),
        ("suffix", "asuffix", True, 1),
        # leading/trailing whitespace
        (" hello", "hello", True, 1),
        ("hello", "hello ", True, 1),
        # long almost-equal string
        (
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy cog",
            True,
            1,
        ),
        (
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy cog",
            False,
            1,
        ),
        # both empty
        ("", "", True, 0),
        ("", "", False, 0),
        # mixed case with symbols
        ("123ABC!", "123abc!", True, 3),
        ("123ABC!", "123abc!", False, 0),
    ],
)
def test_edit_distance(string_1, string_2, case_sensitive, expected_result):
    assert (
        edit_distance(string_1, string_2, case_sensitive=case_sensitive)
        == expected_result
    )


@pytest.mark.parametrize(
    "candidates, query, threshold, case_sensitive, expected_result",
    [
        # exact match at index 0
        (["cat", "dog", "rat"], "cat", 0, True, 0),
        # match at index 2 within threshold
        (["cat", "dog", "rat"], "dat", 1, True, 0),
        # no match due to high threshold
        (["cat", "dog", "rat"], "bat", 0, True, None),
        # multiple possible matches, returns first
        (["apple", "apply", "appla"], "apple", 1, True, 0),
        # case-insensitive match
        (["Alpha", "beta", "Gamma"], "alpha", 0, False, 0),
        # case-sensitive: no match
        (["Alpha", "beta", "Gamma"], "alpha", 0, True, None),
        # threshold boundary
        (["alpha", "beta", "gamma"], "bata", 1, True, 1),
        # no match (all distances too high)
        (["one", "two", "three"], "ten", 1, True, None),
        # unicode/emoji match
        (["😊", "😢", "😁"], "😄", 1, True, 0),
        (["😊", "😢", "😁"], "😊", 0, True, 0),
        # empty candidates
        ([], "any", 2, True, None),
        # empty query, non-empty candidates
        (["", "abc"], "", 0, True, 0),
        (["", "abc"], "", 1, True, 0),
        (["a", "b", "c"], "", 1, True, 0),
        # non-empty query, empty candidate
        (["", ""], "a", 1, True, 0),
        # all candidates require higher edit than threshold
        (["short", "words", "only"], "longerword", 2, True, None),
        # repeated candidates
        (["a", "a", "a"], "b", 1, True, 0),
    ],
)
def test_fuzzy_match_index(
    candidates, query, threshold, case_sensitive, expected_result
):
    assert (
        fuzzy_match_index(
            candidates=candidates,
            query=query,
            threshold=threshold,
            case_sensitive=case_sensitive,
        )
        == expected_result
    )
