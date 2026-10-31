"""Generate offline benchmark JSONL task fixtures for rl-sandbox."""

from __future__ import annotations

import json
import pathlib


def main() -> None:
    tasks_dir = pathlib.Path("src/rl_sandbox/benchmarks/tasks")
    tasks_dir.mkdir(parents=True, exist_ok=True)

    # Define 30 benchmark tasks with 10 candidates each (~300 samples total)
    task_types = [
        (
            "fibonacci",
            "def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
            "assert fib(5) == 5; assert fib(0) == 0",
        ),
        (
            "factorial",
            "def fact(n):\n    return 1 if n == 0 else n * fact(n-1)",
            "assert fact(5) == 120; assert fact(0) == 1",
        ),
        (
            "is_prime",
            "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
            "assert is_prime(7) is True; assert is_prime(4) is False",
        ),
        (
            "reverse_string",
            "def rev(s):\n    return s[::-1]",
            "assert rev('hello') == 'olleh'",
        ),
        (
            "sum_list",
            "def sum_list(lst):\n    return sum(lst)",
            "assert sum_list([1, 2, 3]) == 6",
        ),
        (
            "max_element",
            "def get_max(lst):\n    return max(lst) if lst else None",
            "assert get_max([1, 5, 3]) == 5",
        ),
        (
            "is_palindrome",
            "def is_pal(s):\n    return s == s[::-1]",
            "assert is_pal('racecar') is True",
        ),
        (
            "count_vowels",
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')",
            "assert count_vowels('hello') == 2",
        ),
        (
            "gcd",
            "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a",
            "assert gcd(48, 18) == 6",
        ),
        (
            "binary_search",
            "def bs(arr, target):\n    l, r = 0, len(arr)-1\n    while l <= r:\n        m = (l+r)//2\n        if arr[m] == target: return m\n        elif arr[m] < target: l = m + 1\n        else: r = m - 1\n    return -1",
            "assert bs([1, 2, 3, 4], 3) == 2",
        ),
        (
            "bubble_sort",
            "def bubble(arr):\n    a = list(arr)\n    for i in range(len(a)):\n        for j in range(len(a)-i-1):\n            if a[j] > a[j+1]: a[j], a[j+1] = a[j+1], a[j]\n    return a",
            "assert bubble([3, 1, 2]) == [1, 2, 3]",
        ),
        (
            "flatten",
            "def flatten(lst):\n    res = []\n    for i in lst:\n        if isinstance(i, list): res.extend(flatten(i))\n        else: res.append(i)\n    return res",
            "assert flatten([1, [2, 3], [4]]) == [1, 2, 3, 4]",
        ),
        (
            "merge_dicts",
            "def merge(d1, d2):\n    return {**d1, **d2}",
            "assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}",
        ),
        (
            "remove_duplicates",
            "def dedup(lst):\n    return list(dict.fromkeys(lst))",
            "assert dedup([1, 2, 2, 3]) == [1, 2, 3]",
        ),
        (
            "is_anagram",
            "def is_ana(s1, s2):\n    return sorted(s1) == sorted(s2)",
            "assert is_ana('listen', 'silent') is True",
        ),
        (
            "power",
            "def power(base, exp):\n    return base ** exp",
            "assert power(2, 3) == 8",
        ),
        (
            "clamp",
            "def clamp(val, low, high):\n    return max(low, min(val, high))",
            "assert clamp(5, 0, 10) == 5; assert clamp(-1, 0, 10) == 0",
        ),
        (
            "chunk",
            "def chunk(lst, size):\n    return [lst[i:i + size] for i in range(0, len(lst), size)]",
            "assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
        ),
        (
            "intersection",
            "def intersect(l1, l2):\n    return list(set(l1) & set(l2))",
            "assert set(intersect([1, 2], [2, 3])) == {2}",
        ),
        (
            "running_sum",
            "def run_sum(lst):\n    res, s = [], 0\n    for x in lst: s += x; res.append(s)\n    return res",
            "assert run_sum([1, 2, 3]) == [1, 3, 6]",
        ),
        (
            "is_sorted",
            "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))",
            "assert is_sorted([1, 2, 3]) is True",
        ),
        (
            "frequency",
            "def freq(lst):\n    res = {}\n    for x in lst: res[x] = res.get(x, 0) + 1\n    return res",
            "assert freq([1, 1, 2]) == {1: 2, 2: 1}",
        ),
        (
            "zip_dict",
            "def zip_d(k, v):\n    return dict(zip(k, v))",
            "assert zip_d(['a'], [1]) == {'a': 1}",
        ),
        (
            "longest_word",
            "def long_word(s):\n    words = s.split()\n    return max(words, key=len) if words else ''",
            "assert long_word('cat elephant') == 'elephant'",
        ),
        (
            "to_camel_case",
            "def camel(s):\n    parts = s.split('_')\n    return parts[0] + ''.join(p.capitalize() for p in parts[1:])",
            "assert camel('foo_bar') == 'fooBar'",
        ),
        (
            "is_subsequence",
            "def is_sub(s, t):\n    it = iter(t)\n    return all(c in it for c in s)",
            "assert is_sub('ace', 'abcde') is True",
        ),
        (
            "count_words",
            "def count_w(s):\n    return len(s.split())",
            "assert count_w('hello world') == 2",
        ),
        (
            "median",
            "def median(lst):\n    s = sorted(lst)\n    n = len(s)\n    if not n: return None\n    return s[n//2] if n % 2 != 0 else (s[n//2 - 1] + s[n//2]) / 2",
            "assert median([1, 3, 2]) == 2",
        ),
        (
            "unique_chars",
            "def unique(s):\n    return len(set(s)) == len(s)",
            "assert unique('abc') is True; assert unique('aba') is False",
        ),
        (
            "common_prefix",
            "def prefix(strs):\n    if not strs: return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p):\n            p = p[:-1]\n            if not p: return ''\n    return p",
            "assert prefix(['flower', 'flow']) == 'flow'",
        ),
    ]

    total_records = 0
    for idx, (task_name, correct_code, assertions) in enumerate(task_types):
        task_id = f"task-{idx + 1:02d}-{task_name}"
        records = []

        # 1-6: Correct solutions (various styles)
        for c_idx in range(1, 7):
            records.append(
                {
                    "submission": {
                        "code": f"# candidate {c_idx}\n{correct_code}",
                        "task_id": task_id,
                        "candidate_id": f"cand-{c_idx:02d}",
                        "timeout_seconds": 10.0,
                    },
                    "harness": {"assertions": assertions},
                }
            )

        # 7-8: Logic error candidates
        for c_idx in range(7, 9):
            records.append(
                {
                    "submission": {
                        "code": f"# faulty candidate {c_idx}\nx = 0",
                        "task_id": task_id,
                        "candidate_id": f"cand-{c_idx:02d}",
                        "timeout_seconds": 10.0,
                    },
                    "harness": {"assertions": assertions},
                }
            )

        # 9: Timeout / Infinite loop candidate
        records.append(
            {
                "submission": {
                    "code": "# infinite loop candidate\nwhile True: pass",
                    "task_id": task_id,
                    "candidate_id": "cand-09",
                    "timeout_seconds": 1.0,
                },
                "harness": {"assertions": assertions},
            }
        )

        # 10: Security violation candidate
        records.append(
            {
                "submission": {
                    "code": "# security violation candidate\nimport os; os.system('echo hack')",
                    "task_id": task_id,
                    "candidate_id": "cand-10",
                    "timeout_seconds": 10.0,
                },
                "harness": {"assertions": assertions},
            }
        )

        fixture_file = tasks_dir / f"{task_id}.jsonl"
        with fixture_file.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
                total_records += 1

    print(
        f"Generated {len(task_types)} task files with {total_records} total benchmark samples under {tasks_dir}"
    )


if __name__ == "__main__":
    main()
