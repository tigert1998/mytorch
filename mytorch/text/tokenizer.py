import heapq
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import regex as re
from functools import lru_cache
import json

from tqdm import tqdm


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


class Merge:
    idx0: int
    idx1: int
    __slots__ = ["idx0", "idx1"]

    def __init__(self, idx0, idx1):
        self.idx0 = idx0
        self.idx1 = idx1

    def __hash__(self):
        return self.idx0 * 1000000007 + self.idx1

    def __eq__(self, other: "Merge"):
        return self.idx0 == other.idx0 and self.idx1 == other.idx1


class TrainHeapItem:
    merge: Merge
    count: int
    vocab: Dict[int, str]
    __slots__ = ["merge", "count", "vocab"]

    def __init__(self, merge, count, vocab):
        self.merge = merge
        self.count = count
        self.vocab = vocab

    def __lt__(self, other: "TrainHeapItem"):
        return (
            self.count > other.count
            or self.count == other.count
            and self.vocab[self.merge.idx0] > self.vocab[other.merge.idx0]
            or self.count == other.count
            and self.vocab[self.merge.idx0] == self.vocab[other.merge.idx0]
            and self.vocab[self.merge.idx1] > self.vocab[other.merge.idx1]
        )


class Node:
    l: "Node"
    r: "Node"
    idx: int
    __slots__ = ["l", "r", "idx"]


class BPETokenizer:
    GPT2_PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    vocab: Dict[int, bytes]
    merges: Dict[int, Merge]
    _vocab_dic: Dict[bytes, int]
    special_tokens: List[str]
    _special_token_pattern: str

    def __init__(
        self,
        vocab: Optional[Dict[int, bytes]] = None,
        merges: Optional[Dict[int, Merge]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = special_tokens if special_tokens is not None else []
        if len(self.special_tokens) > 0:
            self._special_token_pattern = self.build_special_token_pattern(
                self.special_tokens
            )
        else:
            self._special_token_pattern = None

        self.vocab = vocab if vocab is not None else {}
        self.merges = merges if merges is not None else {}
        self._vocab_dic = {v: k for k, v in self.vocab.items()}

    def _pre_tokenize(
        self, text_array: List[str], vocab_dic: Dict[bytes, int], train: bool
    ) -> List[List[int]]:
        groups = []
        for text in text_array:
            if not text:
                continue
            if text in self.special_tokens:
                if train:
                    continue
                else:
                    groups.append([vocab_dic[text.encode("utf-8")]])
            else:
                indices_array: List[List[int]] = [
                    [vocab_dic[bytes([b])] for b in s.encode("utf-8")]
                    for s in re.findall(self.GPT2_PATTERN, text)
                ]
                groups.extend(indices_array)
        return groups

    def _internal_encode(self, text: str) -> List[int]:
        if self._special_token_pattern is not None:
            text_array = re.split(self._special_token_pattern, text)
        else:
            text_array = [text]
        groups = self._pre_tokenize(text_array, self._vocab_dic, False)

        merge_to_nodes: Dict[Merge, Dict[Node, None]] = defaultdict(dict)
        num_nodes = sum([len(g) for g in groups])
        nodes = [Node() for _ in range(num_nodes)]
        groups_offset = [None for _ in range(len(groups))]
        offset = 0
        for group_id, g in enumerate(groups):
            groups_offset[group_id] = offset
            for i, idx in enumerate(g):
                nodes[offset + i].idx = idx
                nodes[offset + i].l = None if i == 0 else nodes[offset + i - 1]
                nodes[offset + i].r = None if i + 1 >= len(g) else nodes[offset + i + 1]
                if i + 1 < len(g):
                    merge = Merge(idx, g[i + 1])
                    merge_to_nodes[merge][nodes[offset + i]] = None
            offset += len(g)

        for new_idx in sorted(self.merges.keys()):
            current_merge = self.merges[new_idx]
            ls = list(merge_to_nodes[current_merge].keys())
            for node in ls:
                if not (
                    node.r is not None
                    and node.idx == current_merge.idx0
                    and node.r.idx == current_merge.idx1
                ):
                    continue
                if node.l is not None:
                    merge = Merge(node.l.idx, node.idx)
                    del merge_to_nodes[merge][node.l]
                    merge = Merge(node.l.idx, new_idx)
                    merge_to_nodes[merge][node.l] = None
                if node.r.r is not None:
                    merge = Merge(node.r.idx, node.r.r.idx)
                    del merge_to_nodes[merge][node.r]
                    merge = Merge(new_idx, node.r.r.idx)
                    merge_to_nodes[merge][node] = None
                r = node.r
                node.r = r.r
                if r.r is not None:
                    r.r.l = node
                node.idx = new_idx
                r.idx = -1

        indices = []
        for group_id in range(len(groups)):
            node = nodes[groups_offset[group_id]]
            while node is not None:
                indices.append(node.idx)
                node = node.r
        return indices

    def _internal_decode(self, indices: List[int]):
        return b"".join(self.vocab[idx] for idx in indices).decode(
            "utf-8", errors="replace"
        )

    @staticmethod
    def from_file(file_path) -> "BPETokenizer":
        s2i = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.loads(f.read())
        special_tokens = obj["special_tokens"]
        vocab = obj["vocab"]
        merges = obj["merges"]
        tokenizer_vocab = {k: bytes([s2i[i] for i in v]) for k, v in enumerate(vocab)}
        tokenizer_vocab_dic = {v: k for k, v in tokenizer_vocab.items()}
        tokenizer_merges = {}
        for s in merges:
            s1, s2 = s.split(" ")
            b1 = bytes([s2i[i] for i in s1])
            b2 = bytes([s2i[i] for i in s2])
            merge = Merge(tokenizer_vocab_dic[b1], tokenizer_vocab_dic[b2])
            idx = tokenizer_vocab_dic[b1 + b2]
            tokenizer_merges[idx] = merge
        return BPETokenizer(tokenizer_vocab, tokenizer_merges, special_tokens)

    def save(self, path):
        i2s = gpt2_bytes_to_unicode()

        vocab = [None] * (max(self.vocab.keys()) + 1)
        for k, v in self.vocab.items():
            vocab[k] = "".join([i2s[i] for i in v])
        merges = [
            "".join([i2s[i] for i in self.vocab[self.merges[k].idx0]])
            + " "
            + "".join([i2s[i] for i in self.vocab[self.merges[k].idx1]])
            for k in sorted(self.merges.keys())
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "special_tokens": self.special_tokens,
                    "vocab": vocab,
                    "merges": merges,
                },
                f,
                ensure_ascii=False,
            )

    @staticmethod
    def build_special_token_pattern(special_tokens):
        return f"({'|'.join(re.escape(s) for s in sorted(special_tokens, key=lambda s: -len(s)))})"

    def train(
        self, text: str, vocab_size: int, special_tokens: Optional[List[str]] = None
    ):
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self._special_token_pattern = self.build_special_token_pattern(
                self.special_tokens
            )

        if self._special_token_pattern is not None:
            text_array = re.split(self._special_token_pattern, text)
            initial_vocab = [s.encode("utf-8") for s in self.special_tokens]
        else:
            text_array = [text]
            initial_vocab = []
        for i in range(256):
            initial_vocab.append(bytes([i]))
        initial_vocab_dic = {v: k for k, v in enumerate(initial_vocab)}
        groups = self._pre_tokenize(text_array, initial_vocab_dic, True)

        self._merge(groups, vocab_size - len(initial_vocab), initial_vocab)

    def _merge(self, groups, num_merges, initial_vocab):
        num_nodes = sum([len(g) for g in groups])
        nodes = [Node() for _ in range(num_nodes)]
        freq_table: Dict[Merge, int] = defaultdict(int)
        vocab = [None for _ in range(len(initial_vocab) + num_merges)]
        merges = [None for _ in range(num_merges)]
        for i in range(len(initial_vocab)):
            vocab[i] = initial_vocab[i]
        merge_to_nodes: Dict[Merge, Dict[Node, None]] = defaultdict(dict)
        offset = 0
        for g in groups:
            for i, idx in enumerate(g):
                nodes[offset + i].idx = idx
                nodes[offset + i].l = None if i == 0 else nodes[offset + i - 1]
                nodes[offset + i].r = None if i + 1 >= len(g) else nodes[offset + i + 1]
                if i + 1 < len(g):
                    merge = Merge(idx, g[i + 1])
                    freq_table[merge] += 1
                    merge_to_nodes[merge][nodes[offset + i]] = None
            offset += len(g)
        heap = [TrainHeapItem(k, v, vocab) for k, v in freq_table.items()]
        heapq.heapify(heap)
        for merge_id in tqdm(range(num_merges)):
            while True:
                top = heapq.heappop(heap)
                if freq_table.get(top.merge) is None:
                    continue
                if freq_table[top.merge] != top.count:
                    heapq.heappush(
                        heap, TrainHeapItem(top.merge, freq_table[top.merge], vocab)
                    )
                    continue
                break
            merges[merge_id] = top.merge
            new_idx = merge_id + len(initial_vocab)
            vocab[new_idx] = vocab[top.merge.idx0] + vocab[top.merge.idx1]
            for node in list(merge_to_nodes[top.merge].keys()):
                if not (
                    node.r is not None
                    and node.idx == top.merge.idx0
                    and node.r.idx == top.merge.idx1
                ):
                    # prevent phantom node
                    continue
                if node.l is not None:
                    merge = Merge(node.l.idx, node.idx)
                    del merge_to_nodes[merge][node.l]
                    freq_table[merge] -= 1
                    merge = Merge(node.l.idx, new_idx)
                    merge_to_nodes[merge][node.l] = None
                    freq_table[merge] += 1
                    heapq.heappush(heap, TrainHeapItem(merge, freq_table[merge], vocab))
                if node.r.r is not None:
                    merge = Merge(node.r.idx, node.r.r.idx)
                    del merge_to_nodes[merge][node.r]
                    freq_table[merge] -= 1
                    merge = Merge(new_idx, node.r.r.idx)
                    merge_to_nodes[merge][node] = None
                    freq_table[merge] += 1
                    heapq.heappush(heap, TrainHeapItem(merge, freq_table[merge], vocab))
                r = node.r
                node.r = r.r
                if r.r is not None:
                    r.r.l = node
                node.idx = new_idx
                r.idx = -1
            del freq_table[top.merge]
            del merge_to_nodes[top.merge]

        self.vocab = {k: v for k, v in enumerate(vocab)}
        self._vocab_dic = {v: k for k, v in enumerate(self.vocab)}
        self.merges = {i + len(initial_vocab): merge for i, merge in enumerate(merges)}
