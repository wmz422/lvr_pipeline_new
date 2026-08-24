"""答案抽取（多选题打分）。"""

from __future__ import annotations

import re
import string


def parse_choices_from_text(text: str) -> list[str]:
    """从 benchmark question/options 文本里抽出选项内容。

    支持 VSTAR 的多行 ``(A) foo`` 和 MMVP 的同行 ``(a) Foo (b) Bar``。
    返回值只含选项文本，顺序对应 A/B/C...
    """
    if not text:
        return []

    matches = list(re.finditer(r"\(([A-Za-z])\)\s*", text))
    if not matches:
        return []

    choices: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        value = re.sub(r"\s*Answer with the option.*$", "", value, flags=re.IGNORECASE | re.DOTALL)
        value = value.strip(" \t\r\n.;,")
        if value:
            choices.append(value)
    return choices


def _allowed_letters(choices: list[str] | None) -> set[str]:
    if choices:
        return set(string.ascii_uppercase[: len(choices)])
    return set("ABCD")


def _valid(letter: str, allowed: set[str]) -> str:
    letter = letter.upper()
    return letter if letter in allowed else ""


def _last_valid(pattern: str, text: str, allowed: set[str], flags: int = re.IGNORECASE) -> str:
    matches = list(re.finditer(pattern, text, flags))
    for match in reversed(matches):
        letter = _valid(match.group(1), allowed)
        if letter:
            return letter
    return ""


def _match_choice_by_content(tails: list[str], choices: list[str] | None, allowed: set[str]) -> str:
    """模型直接答出选项**内容**（如 "No"、"Open"）而非字母时，把内容匹配回字母。

    MMVP 这类题模型常只输出 "No"/"Yes"/"Open"，字母策略全失败。仅作兜底，且要求唯一命中避免歧义。
    """
    if not choices:
        return ""

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    norm_choices = [(string.ascii_uppercase[i], norm(c)) for i, c in enumerate(choices)]
    for tail in tails:
        ans = norm(tail)
        if not ans:
            continue
        # 1) 答案与某选项规范化后完全相等。
        for letter, c in norm_choices:
            if c and c == ans and letter in allowed:
                return letter
        # 2) 答案里整词命中某选项，且只命中一个（去歧义）。
        hits: list[str] = []
        for letter, c in norm_choices:
            if c and letter in allowed and re.search(r"\b" + re.escape(c) + r"\b", ans):
                if letter not in hits:
                    hits.append(letter)
        if len(hits) == 1:
            return hits[0]
    return ""


def extract_answer(text: str, choices: list[str] | None = None) -> str:
    """从模型输出里抽取答案字母（A/B/C/D...）。多策略，按置信度排序，首个命中即返回：

    1. <answer>X</answer> 标签（含少闭合 `<answer>B`）—— 精确匹配
    2. "the answer is X" / "option X" / 中文"答案是X" 等模式
    3. 答案区（`</think>` 后 / 末行 / ≤80 字符全文）里的独立合法字母，含 `(A)`、`A.`
    4. 模型直接答选项**内容**（"No"/"Open"…）→ 按内容匹配回字母（唯一命中）

    全程只接受 `choices` 数量内的字母（allowed），多处命中取**最后**一个。都不中返回 ""。
    """
    if not text or not isinstance(text, str):
        return ""

    text_clean = text.strip()
    # 去掉 latent special token（<abs_vis_token> / </abs_vis_token> / <abs_vis_token_pad>），避免干扰答案区识别。
    text_clean = re.sub(r"</?abs_vis_token(?:_pad)?>", " ", text_clean).strip()
    allowed = _allowed_letters(choices)

    # 策略 1：<answer>X</answer>
    answer_tag = _last_valid(r"<answer>\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*</answer>", text_clean, allowed)
    if answer_tag:
        return answer_tag

    # 兼容少闭合标签的输出，如 "<answer>B"。
    answer_tag_open = _last_valid(r"<answer>\s*[\(\[]?\s*([A-Za-z])\b", text_clean, allowed)
    if answer_tag_open:
        return answer_tag_open

    # 策略 2："the answer is X" / "answer should be X" / "option X" 等。
    patterns = [
        r"(?:final\s+answer|correct\s+answer|the\s+answer|answer)\s*"
        r"(?:should|would|must|could|seems\s+to)?\s*(?:be|is|:)?\s*"
        r"(?:option|choice)?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?",
        r"(?:option|choice)\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?",
        r"(?:答案|最终答案|正确答案)\s*(?:应该是|是|为|:|：)?\s*(?:选项)?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?",
    ]
    for pat in patterns:
        letter = _last_valid(pat, text_clean, allowed)
        if letter:
            return letter

    # 策略 3：只在明确的最终答案区接受裸字母。长 CoT 被 max_new_tokens 截断时，
    # 全文兜底会从 "A-D" / "Options" 里捞出随机字母，这里避免这种误判。
    candidate_tails: list[str] = []
    if "</think>" in text_clean:
        candidate_tails.append(text_clean.rsplit("</think>", 1)[-1].strip())
    candidate_tails.append(text_clean.rsplit("\n", 1)[-1].strip())
    if len(text_clean) <= 80:
        candidate_tails.append(text_clean)

    for tail in candidate_tails:
        if not tail or len(tail) > 120:
            continue
        standalone_tail = _last_valid(r"\b([A-Za-z])\b", tail, allowed, flags=0)
        if standalone_tail:
            return standalone_tail
        # 形如 "(A)"、"A."、"B)" 的也算。
        letter_punct_tail = _last_valid(r"[\(\[]?\b([A-Za-z])\b\s*[\)\]\.]", tail, allowed, flags=0)
        if letter_punct_tail:
            return letter_punct_tail

    # 策略 4：模型直接答选项内容（"No"/"Open"…）而非字母 → 按内容匹配回字母。
    content_letter = _match_choice_by_content(candidate_tails, choices, allowed)
    if content_letter:
        return content_letter

    return ""
