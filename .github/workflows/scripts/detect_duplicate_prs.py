# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Detect duplicate PRs using text similarity + file overlap.
"""

import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from sklearn.feature_extraction.text import HashingVectorizer

USE_SENTENCE_TRANSFORMERS = os.getenv("USE_SENTENCE_TRANSFORMERS", "1").lower() in {
    "1",
    "true",
    "yes",
}
try:
    if USE_SENTENCE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer
    else:
        SentenceTransformer = None
except Exception:
    SentenceTransformer = None

model = None
if SentenceTransformer is not None:
    model = SentenceTransformer("all-MiniLM-L6-v2")

hashing_vectorizer = HashingVectorizer(
    n_features=2048,
    alternate_sign=False,
    norm="l2",
)


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
PR_NUMBER = int(os.environ["PR_NUMBER"])
REPO = os.environ["REPO"]
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SIMILARITY_THRESHOLD = 0.82
SOFT_SIMILARITY_THRESHOLD = float(os.getenv("SOFT_SIMILARITY_THRESHOLD", "0.72"))
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_OPEN_CANDIDATES = int(os.getenv("MAX_OPEN_CANDIDATES", "120"))
FILE_COMPARE_TOP_N = int(os.getenv("FILE_COMPARE_TOP_N", "30"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.55"))
FILE_WEIGHT = float(os.getenv("FILE_WEIGHT", "0.20"))
PREFIX_WEIGHT = float(os.getenv("PREFIX_WEIGHT", "0.15"))
TOKEN_WEIGHT = float(os.getenv("TOKEN_WEIGHT", "0.10"))
SAME_AUTHOR_PENALTY = float(os.getenv("SAME_AUTHOR_PENALTY", "0.03"))
PREFIX_DEPTH = int(os.getenv("PREFIX_DEPTH", "2"))
TOKEN_MIN_LEN = int(os.getenv("TOKEN_MIN_LEN", "3"))
TOKEN_MAX_COUNT = int(os.getenv("TOKEN_MAX_COUNT", "40"))
MAX_MERGED_CANDIDATES = int(os.getenv("MAX_MERGED_CANDIDATES", "120"))
RECENT_MERGED_DAYS = int(os.getenv("RECENT_MERGED_DAYS", "180"))


def gh_get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def get_pr_files(pr_number):
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/files"
    try:
        files = gh_get(url, params={"per_page": 100})
        return [f["filename"] for f in files]
    except Exception:
        return []


def build_pr_text(pr):
    parts = [
        f"Title: {pr.get('title', '')}",
        f"Body: {(pr.get('body') or '')[:800]}",
    ]
    return "\n".join(parts)


def get_embedding(text: str):
    if model is not None:
        return np.asarray(model.encode(text), dtype=float)
    return hashing_vectorizer.transform([text]).toarray()[0]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def jaccard_similarity(a, b):
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def path_prefix(path: str, depth: int = PREFIX_DEPTH):
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    return "/".join(parts[:depth])


def path_prefix_similarity(a_paths, b_paths):
    a_prefixes = [path_prefix(p) for p in a_paths]
    b_prefixes = [path_prefix(p) for p in b_paths]
    return jaccard_similarity(a_prefixes, b_prefixes)


def extract_tokens(text: str):
    # Keep alnum/underscore/hyphen tokens to preserve API and module names.
    tokens = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_-]*", (text or "").lower())
    filtered = [t for t in tokens if len(t) >= TOKEN_MIN_LEN and not t.isdigit()]
    return set(filtered[:TOKEN_MAX_COUNT])


def token_similarity(a_text: str, b_text: str):
    return jaccard_similarity(extract_tokens(a_text), extract_tokens(b_text))


def parse_iso_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_recent_merged_prs(days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = []
    page = 1
    while len(result) < MAX_MERGED_CANDIDATES:
        prs = gh_get(
            f"https://api.github.com/repos/{REPO}/pulls",
            params={
                "state": "closed",
                "per_page": 50,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        if not prs:
            break
        reached_cutoff = False
        for pr in prs:
            if pr["number"] == PR_NUMBER:
                continue
            if not pr.get("merged_at"):
                continue
            updated_at = parse_iso_time(pr.get("updated_at"))
            if updated_at is None or updated_at < cutoff:
                reached_cutoff = True
                continue
            result.append(pr)
            if len(result) >= MAX_MERGED_CANDIDATES:
                break
        if len(prs) < 50 or reached_cutoff:
            break
        page += 1
    return result


def post_comment(issue_number, body):
    if DRY_RUN:
        print("DRY_RUN enabled: skip posting PR comment.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=HEADERS, json={"body": body})


def main():
    # 1. Load current PR context.
    current_pr = gh_get(f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}")
    current_text = build_pr_text(current_pr)
    current_emb = get_embedding(current_text)
    current_files = get_pr_files(PR_NUMBER)
    current_author = (current_pr.get("user") or {}).get("login")

    # 2. Fetch open PR candidates.
    open_prs = []
    page = 1
    while len(open_prs) < MAX_OPEN_CANDIDATES:
        prs = gh_get(
            f"https://api.github.com/repos/{REPO}/pulls",
            params={
                "state": "open",
                "per_page": 50,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        if not prs:
            break
        for pr in prs:
            if pr["number"] != PR_NUMBER:
                open_prs.append(pr)
                if len(open_prs) >= MAX_OPEN_CANDIDATES:
                    break
        page += 1
        if len(prs) < 50:
            break
    merged_prs = get_recent_merged_prs(RECENT_MERGED_DAYS)
    seen_numbers = set()
    history_prs = []
    for pr in open_prs + merged_prs:
        if pr["number"] in seen_numbers:
            continue
        seen_numbers.add(pr["number"])
        history_prs.append(pr)

    # 3. Stage-1: text-only retrieval for candidate pruning.
    text_results = []
    for pr in history_prs:
        text = build_pr_text(pr)
        emb = get_embedding(text)
        text_sim = cosine_similarity(current_emb, emb)
        text_results.append((text_sim, pr))

    text_results.sort(key=lambda x: -x[0])
    file_candidates = text_results[:FILE_COMPARE_TOP_N]

    # 4. Stage-2: enrich with file/path/token features and weighted score.
    results = []
    for text_sim, pr in file_candidates:
        pr_files = get_pr_files(pr["number"])
        file_sim = jaccard_similarity(current_files, pr_files)
        prefix_sim = path_prefix_similarity(current_files, pr_files)
        pr_text = build_pr_text(pr)
        token_sim = token_similarity(current_text, pr_text)
        same_author = current_author and current_author == (pr.get("user") or {}).get(
            "login"
        )
        author_penalty = SAME_AUTHOR_PENALTY if same_author else 0.0
        final_sim = (
            TEXT_WEIGHT * text_sim
            + FILE_WEIGHT * file_sim
            + PREFIX_WEIGHT * prefix_sim
            + TOKEN_WEIGHT * token_sim
            - author_penalty
        )
        results.append((final_sim, pr, text_sim, file_sim, prefix_sim, token_sim))

    results.sort(key=lambda x: -x[0])
    top_results = [
        (sim, pr, text_sim, file_sim, prefix_sim, token_sim)
        for sim, pr, text_sim, file_sim, prefix_sim, token_sim in results[:TOP_K]
        if sim >= SIMILARITY_THRESHOLD
        or (
            sim >= SOFT_SIMILARITY_THRESHOLD
            and (prefix_sim >= 0.50 or token_sim >= 0.40)
        )
    ]

    # 5. Post comment when candidates are found.
    if not top_results:
        print("No highly similar PRs found.")
        return
    lines = [
        "## 🔍 Potential Duplicate PRs Detected\n",
        "The following open or recently merged PRs appear similar to this one:\n",
        "| Score | Text | Files | Prefix | Tokens | PR | State | Title |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sim, pr, text_sim, file_sim, prefix_sim, token_sim in top_results:
        state_icon = (
            "🟢" if pr["state"] == "open" else ("🟣" if pr.get("merged_at") else "🔴")
        )
        state_text = "merged" if pr.get("merged_at") else pr["state"]
        row = (
            f"| {sim:.0%} | {text_sim:.0%} | {file_sim:.0%} | {prefix_sim:.0%} | "
            f"{token_sim:.0%} | #{pr['number']} | {state_icon} {state_text} | "
            f"[{pr['title']}]({pr['html_url']}) |"
        )
        lines.append(row)
    lines.append("\n> 🤖 Auto-detected by duplicate PR checker.")
    lines.append("Please review to avoid redundant work.")
    post_comment(PR_NUMBER, "\n".join(lines))
    print(f"Posted comment with {len(top_results)} similar PRs.")


if __name__ == "__main__":
    main()
