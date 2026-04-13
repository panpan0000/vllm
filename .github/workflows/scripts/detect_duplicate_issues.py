# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Detect duplicate Issues using OpenAI embeddings + cosine similarity.
Compares: title + body keywords.
"""

import os

import numpy as np
import requests

# 替换 get_embedding 函数，使用 sentence-transformers（免费）
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])
REPO = os.environ["REPO"]
DRY_RUN = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SIMILARITY_THRESHOLD = 0.88
TOP_K = 5
MAX_HISTORY = 300


def gh_get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def build_issue_text(issue):
    return f"Title: {issue.get('title', '')}\nBody: {(issue.get('body') or '')[:1000]}"


def get_embedding(text: str):
    return model.encode(text)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def post_comment(issue_number, body):
    if DRY_RUN:
        print("DRY_RUN enabled: skip posting issue comment.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=HEADERS, json={"body": body})


def add_label(issue_number, label):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip adding label {label}.")
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/labels"
    requests.post(url, headers=HEADERS, json={"labels": [label]})


def main():
    # 1. 获取当前 Issue
    current = gh_get(f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}")
    # 跳过 PR（GitHub API 中 PR 也会出现在 issues 接口）
    if "pull_request" in current:
        print("This is a PR, skipping.")
        return

    current_text = build_issue_text(current)
    current_emb = get_embedding(current_text)

    # 2. 拉取历史 Issues（open + closed）
    history_issues = []
    for state in ["open", "closed"]:
        page = 1
        while len(history_issues) < MAX_HISTORY:
            issues = gh_get(
                f"https://api.github.com/repos/{REPO}/issues",
                params={
                    "state": state,
                    "per_page": 50,
                    "page": page,
                    "sort": "updated",
                    "direction": "desc",
                },
            )
            if not issues:
                break
            for issue in issues:
                # 跳过 PR 和自身
                if "pull_request" not in issue and issue["number"] != ISSUE_NUMBER:
                    history_issues.append(issue)
            page += 1
            if len(issues) < 50:
                break
        if len(history_issues) >= MAX_HISTORY:
            break

    history_issues = history_issues[:MAX_HISTORY]

    # 3. 计算相似度
    results = []
    for issue in history_issues:
        text = build_issue_text(issue)
        emb = get_embedding(text)
        sim = cosine_similarity(current_emb, emb)
        results.append((sim, issue))

    results.sort(key=lambda x: -x[0])
    top_results = [
        (sim, i) for sim, i in results[:TOP_K] if sim >= SIMILARITY_THRESHOLD
    ]

    if not top_results:
        print("No highly similar issues found.")
        return

    # 4. 自动打 label + 发评论
    add_label(ISSUE_NUMBER, "possible-duplicate")

    lines = [
        "## 🔍 Possible Duplicate Issue Detected\n",
        "The following existing issues appear highly similar:\n",
        "| Similarity | Issue | State | Title |",
        "|---|---|---|---|",
    ]
    for sim, issue in top_results:
        state_icon = "🟢" if issue["state"] == "open" else "🔴"
        row = (
            f"| {sim:.0%} | #{issue['number']} | {state_icon} {issue['state']} | "
            f"[{issue['title']}]({issue['html_url']}) |"
        )
        lines.append(row)
    lines.append(
        "\n> 🤖 Auto-detected by duplicate issue checker. A maintainer will verify."
    )
    post_comment(ISSUE_NUMBER, "\n".join(lines))
    print(f"Posted comment with {len(top_results)} similar issues.")


if __name__ == "__main__":
    main()
