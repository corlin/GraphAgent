# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the 'Tokenizer', 'TextSplitter', 'NoopTextSplitter' and 'TokenTextSplitter' models."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import pandas as pd
import tiktoken
import jieba

import graphrag.config.defaults as defs
from graphrag.index.operations.chunk_text.typing import TextChunk
from graphrag.logger.progress import ProgressTicker

EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]
LengthFn = Callable[[str], int]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: DecodeFn
    """ Function to decode a list of token ids to a string"""
    encode: EncodeFn
    """ Function to encode a string to a list of token ids"""


class TextSplitter(ABC):
    """Text splitter class definition."""

    _chunk_size: int
    _chunk_overlap: int
    _length_function: LengthFn
    _keep_separator: bool
    _add_start_index: bool
    _strip_whitespace: bool

    def __init__(
        self,
        # based on text-ada-002-embedding max input buffer length
        # https://platform.openai.com/docs/guides/embeddings/second-generation-models
        chunk_size: int = 8191,
        chunk_overlap: int = 100,
        length_function: LengthFn = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ):
        """Init method definition."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""


class NoopTextSplitter(TextSplitter):
    """Noop text splitter class definition."""

    def split_text(self, text: str | list[str]) -> Iterable[str]:
        """Split text method definition."""
        return [text] if isinstance(text, str) else text


class TokenTextSplitter(TextSplitter):
    """Token text splitter class definition."""

    _allowed_special: Literal["all"] | set[str]
    _disallowed_special: Literal["all"] | Collection[str]

    def __init__(
        self,
        encoding_name: str = defs.ENCODING_MODEL,
        model_name: str | None = None,
        allowed_special: Literal["all"] | set[str] | None = None,
        disallowed_special: Literal["all"] | Collection[str] = "all",
        **kwargs: Any,
    ):
        """Init method definition."""
        super().__init__(**kwargs)
        if model_name is not None:
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                log.exception("Model %s not found, using %s", model_name, encoding_name)
                enc = tiktoken.get_encoding(encoding_name)
        else:
            enc = tiktoken.get_encoding(encoding_name)
        self._tokenizer = enc
        self._allowed_special = allowed_special or set()
        self._disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        """Encode the given text into an int-vector."""
        return self._tokenizer.encode(
            text,
            allowed_special=self._allowed_special,
            disallowed_special=self._disallowed_special,
        )

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self.encode(text))

    def split_text(self, text: str | list[str]) -> list[str]:
        """Split text method."""
        if isinstance(text, list):
            text = " ".join(text)
        elif cast("bool", pd.isna(text)) or text == "":
            return []
        if not isinstance(text, str):
            msg = f"Attempting to split a non-string value, actual is {type(text)}"
            raise TypeError(msg)

        tokenizer = Tokenizer(
            chunk_overlap=self._chunk_overlap,
            tokens_per_chunk=self._chunk_size,
            decode=self._tokenizer.decode,
            encode=lambda text: self.encode(text),
        )

        return split_single_text_on_tokens(text=text, tokenizer=tokenizer)


def split_single_text_on_tokens(text: str, tokenizer: Tokenizer) -> list[str]:
    """Split a single text and return chunks using the tokenizer, with comprehensive Chinese text support using jieba."""
    result = []
    input_ids = tokenizer.encode(text)

    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = tokenizer.decode(list(chunk_ids))
        if cur_idx < len(input_ids):
            last_char = chunk_text[-1] if chunk_text else ''
            next_char = tokenizer.decode(input_ids[cur_idx:cur_idx+1]) if cur_idx < len(input_ids) else ''
            if _is_chinese_char(last_char) or _is_chinese_char(next_char):
                # 使用 jieba 分词并考虑句子完整性来找到更准确的中文分割点
                temp_chunk_text = tokenizer.decode(input_ids[start_idx:cur_idx + tokenizer.chunk_overlap])
                words = list(jieba.cut(temp_chunk_text, cut_all=False))
                if words:
                    safe_cut_point = 0
                    sentence_end_markers = ['。', '！', '？', '；', ';']
                    last_sentence_end = 0
                    for i, word in enumerate(words):
                        safe_cut_point += len(word)
                        if any(marker in word for marker in sentence_end_markers):
                            last_sentence_end = safe_cut_point
                        if safe_cut_point >= len(chunk_text):
                            break
                    # 优先在句子结束处分割，如果没有找到合适的句子结束，则回退到词边界
                    if last_sentence_end > 0 and last_sentence_end <= len(chunk_text):
                        safe_cut_point = last_sentence_end
                    # 考虑 tokens_per_chunk 和 chunk_size 的影响，动态调整分割点
                    token_count = len(tokenizer.encode(temp_chunk_text[:safe_cut_point]))
                    if token_count > tokenizer.tokens_per_chunk * 0.9:
                        # 如果接近 tokens_per_chunk 上限，尝试找到更早的句子结束点
                        for i in range(len(words) - 1, -1, -1):
                            temp_safe_cut_point = sum(len(w) for w in words[:i+1])
                            if any(marker in words[i] for marker in sentence_end_markers) and len(tokenizer.encode(temp_chunk_text[:temp_safe_cut_point])) <= tokenizer.tokens_per_chunk * 0.8:
                                safe_cut_point = temp_safe_cut_point
                                break
                    if safe_cut_point < len(temp_chunk_text):
                        cur_idx = start_idx + len(tokenizer.encode(temp_chunk_text[:safe_cut_point]))
                        chunk_ids = input_ids[start_idx:cur_idx]
                        chunk_text = tokenizer.decode(list(chunk_ids))
            # 确保块开始处的字符完整性
            if start_idx > 0:
                start_text = tokenizer.decode(input_ids[start_idx:start_idx+20]) if start_idx+20 < len(input_ids) else tokenizer.decode(input_ids[start_idx:])
                start_words = list(jieba.cut(start_text, cut_all=False))
                if start_words and _is_chinese_char(start_text[0]):
                    # 如果开始处是一个中文字符，检查是否在词语中间
                    cumulative_len = 0
                    for word in start_words:
                        cumulative_len += len(word)
                        if cumulative_len > 0 and _is_chinese_char(word[0]):
                            # 找到第一个完整的中文词语的开始位置
                            for i in range(start_idx - 1, max(0, start_idx - 20), -1):
                                temp_start_text = tokenizer.decode(input_ids[i:start_idx+20])
                                temp_start_words = list(jieba.cut(temp_start_text, cut_all=False))
                                if temp_start_words and not _is_chinese_char(temp_start_text[0]):
                                    start_idx = i
                                    chunk_ids = input_ids[start_idx:cur_idx]
                                    chunk_text = tokenizer.decode(list(chunk_ids))
                                    break
                            break
        result.append(chunk_text)
        if cur_idx == len(input_ids):
            break
        # 确保重叠区域的中文字符完整性
        overlap_start = cur_idx - tokenizer.chunk_overlap if cur_idx - start_idx > tokenizer.chunk_overlap else start_idx
        if overlap_start > 0 and overlap_start < len(input_ids):
            overlap_text = tokenizer.decode(input_ids[overlap_start:overlap_start+10]) if overlap_start+10 < len(input_ids) else tokenizer.decode(input_ids[overlap_start:])
            overlap_words = list(jieba.cut(overlap_text, cut_all=False))
            if overlap_words and len(overlap_words[0]) > 1 and _is_chinese_char(overlap_text[0]):
                # 如果重叠开始处是一个中文词的中间部分，则向前调整overlap_start
                for i in range(overlap_start - 1, max(0, overlap_start - 10), -1):
                    temp_overlap_text = tokenizer.decode(input_ids[i:overlap_start+10])
                    temp_overlap_words = list(jieba.cut(temp_overlap_text, cut_all=False))
                    if temp_overlap_words and not _is_chinese_char(temp_overlap_text[0]):
                        overlap_start = i
                        break
            # 检查重叠区域结束处是否在中文词中间
            overlap_end = min(overlap_start + tokenizer.chunk_overlap, len(input_ids))
            if overlap_end < len(input_ids):
                overlap_end_text = tokenizer.decode(input_ids[overlap_end-10:overlap_end]) if overlap_end-10 > 0 else tokenizer.decode(input_ids[:overlap_end])
                overlap_end_words = list(jieba.cut(overlap_end_text, cut_all=False))
                if overlap_end_words and _is_chinese_char(overlap_end_text[-1]):
                    # 如果重叠结束处是一个中文词的中间部分，则向后调整overlap_end
                    temp_text = tokenizer.decode(input_ids[overlap_end-10:overlap_end+10]) if overlap_end+10 < len(input_ids) else tokenizer.decode(input_ids[overlap_end-10:])
                    temp_words = list(jieba.cut(temp_text, cut_all=False))
                    temp_len = len(overlap_end_text)
                    for word in temp_words:
                        temp_len += len(word)
                        if temp_len > len(overlap_end_text) and not _is_chinese_char(word[-1]):
                            overlap_end = overlap_end + (temp_len - len(overlap_end_text))
                            if overlap_end > len(input_ids):
                                overlap_end = len(input_ids)
                            break
        start_idx = overlap_start
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


def _is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character based on Unicode ranges."""
    # if not char or len(char) != 1:
    #     return False
    # Unicode ranges for Chinese characters
    # CJK Unified Ideographs: 4E00-9FFF
    # CJK Unified Ideographs Extension A: 3400-4DBF
    # CJK Unified Ideographs Extension B: 20000-2A6DF
    try:
        code_point = ord(char)
        return (0x4E00 <= code_point <= 0x9FFF or
                0x3400 <= code_point <= 0x4DBF or
                0x20000 <= code_point <= 0x2A6DF)
    except TypeError:
        return False


# Adapted from - https://github.com/langchain-ai/langchain/blob/77b359edf5df0d37ef0d539f678cf64f5557cb54/libs/langchain/langchain/text_splitter.py#L471
# So we could have better control over the chunking process
def split_multiple_texts_on_tokens(
    texts: list[str], tokenizer: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    """Split multiple texts and return chunks with metadata using the tokenizer, with comprehensive Chinese text support using jieba."""
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        if tick:
            tick(1)
        mapped_ids.append((source_doc_idx, encoded))

    input_ids = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]

    while start_idx < len(input_ids):
        chunk_text = tokenizer.decode([id for _, id in chunk_ids])
        if cur_idx < len(input_ids):
            last_char = chunk_text[-1] if chunk_text else ''
            next_char = tokenizer.decode([id for _, id in input_ids[cur_idx:cur_idx+1]]) if cur_idx < len(input_ids) else ''
            if _is_chinese_char(last_char) or _is_chinese_char(next_char):
                temp_chunk_text = tokenizer.decode([id for _, id in input_ids[start_idx:cur_idx + tokenizer.chunk_overlap]])
                words = list(jieba.cut(temp_chunk_text, cut_all=False))
                if words:
                    safe_cut_point = 0
                    sentence_end_markers = ['。', '！', '？', '；', ';']
                    last_sentence_end = 0
                    for i, word in enumerate(words):
                        safe_cut_point += len(word)
                        if any(marker in word for marker in sentence_end_markers):
                            last_sentence_end = safe_cut_point
                        if safe_cut_point >= len(chunk_text):
                            break
                    # 优先在句子结束处分割，如果没有找到合适的句子结束，则回退到词边界
                    if last_sentence_end > 0 and last_sentence_end <= len(chunk_text):
                        safe_cut_point = last_sentence_end
                    # 考虑 tokens_per_chunk 和 chunk_size 的影响，动态调整分割点
                    token_count = len(tokenizer.encode(temp_chunk_text[:safe_cut_point]))
                    if token_count > tokenizer.tokens_per_chunk * 0.9:
                        # 如果接近 tokens_per_chunk 上限，尝试找到更早的句子结束点
                        for i in range(len(words) - 1, -1, -1):
                            temp_safe_cut_point = sum(len(w) for w in words[:i+1])
                            if any(marker in words[i] for marker in sentence_end_markers) and len(tokenizer.encode(temp_chunk_text[:temp_safe_cut_point])) <= tokenizer.tokens_per_chunk * 0.8:
                                safe_cut_point = temp_safe_cut_point
                                break
                    if safe_cut_point < len(temp_chunk_text):
                        cur_idx = start_idx + len(tokenizer.encode(temp_chunk_text[:safe_cut_point]))
                        chunk_ids = input_ids[start_idx:cur_idx]
                        chunk_text = tokenizer.decode([id for _, id in chunk_ids])
            # 确保块开始处的字符完整性
            if start_idx > 0:
                start_text = tokenizer.decode([id for _, id in input_ids[start_idx:start_idx+20]]) if start_idx+20 < len(input_ids) else tokenizer.decode([id for _, id in input_ids[start_idx:]])
                start_words = list(jieba.cut(start_text, cut_all=False))
                if start_words and _is_chinese_char(start_text[0]):
                    # 如果开始处是一个中文字符，检查是否在词语中间
                    cumulative_len = 0
                    for word in start_words:
                        cumulative_len += len(word)
                        if cumulative_len > 0 and _is_chinese_char(word[0]):
                            # 找到第一个完整的中文词语的开始位置
                            for i in range(start_idx - 1, max(0, start_idx - 20), -1):
                                temp_start_text = tokenizer.decode([id for _, id in input_ids[i:start_idx+20]])
                                temp_start_words = list(jieba.cut(temp_start_text, cut_all=False))
                                if temp_start_words and not _is_chinese_char(temp_start_text[0]):
                                    start_idx = i
                                    chunk_ids = input_ids[start_idx:cur_idx]
                                    chunk_text = tokenizer.decode([id for _, id in chunk_ids])
                                    break
                            break
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(TextChunk(chunk_text, doc_indices, len(chunk_ids)))
        if cur_idx == len(input_ids):
            break
        # 确保重叠区域的中文字符完整性
        overlap_start = cur_idx - tokenizer.chunk_overlap if cur_idx - start_idx > tokenizer.chunk_overlap else start_idx
        if overlap_start > 0 and overlap_start < len(input_ids):
            overlap_text = tokenizer.decode([id for _, id in input_ids[overlap_start:overlap_start+10]]) if overlap_start+10 < len(input_ids) else tokenizer.decode([id for _, id in input_ids[overlap_start:]])
            overlap_words = list(jieba.cut(overlap_text, cut_all=False))
            if overlap_words and len(overlap_words[0]) > 1 and _is_chinese_char(overlap_text[0]):
                # 如果重叠开始处是一个中文词的中间部分，则向前调整overlap_start
                for i in range(overlap_start - 1, max(0, overlap_start - 10), -1):
                    temp_overlap_text = tokenizer.decode([id for _, id in input_ids[i:overlap_start+10]])
                    temp_overlap_words = list(jieba.cut(temp_overlap_text, cut_all=False))
                    if temp_overlap_words and not _is_chinese_char(temp_overlap_text[0]):
                        overlap_start = i
                        break
            # 检查重叠区域结束处是否在中文词中间
            overlap_end = min(overlap_start + tokenizer.chunk_overlap, len(input_ids))
            if overlap_end < len(input_ids):
                overlap_end_text = tokenizer.decode([id for _, id in input_ids[overlap_end-10:overlap_end]]) if overlap_end-10 > 0 else tokenizer.decode([id for _, id in input_ids[:overlap_end]])
                overlap_end_words = list(jieba.cut(overlap_end_text, cut_all=False))
                if overlap_end_words and _is_chinese_char(overlap_end_text[-1]):
                    # 如果重叠结束处是一个中文词的中间部分，则向后调整overlap_end
                    temp_text = tokenizer.decode([id for _, id in input_ids[overlap_end-10:overlap_end+10]]) if overlap_end+10 < len(input_ids) else tokenizer.decode([id for _, id in input_ids[overlap_end-10:]])
                    temp_words = list(jieba.cut(temp_text, cut_all=False))
                    temp_len = len(overlap_end_text)
                    for word in temp_words:
                        temp_len += len(word)
                        if temp_len > len(overlap_end_text) and not _is_chinese_char(word[-1]):
                            overlap_end = overlap_end + (temp_len - len(overlap_end_text))
                            if overlap_end > len(input_ids):
                                overlap_end = len(input_ids)
                            break
        start_idx = overlap_start
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result