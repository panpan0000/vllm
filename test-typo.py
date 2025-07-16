# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Test file containing spelling errors found in commit 9dfe0dceaa4aab0820277da7d24bb081f88608ae
# This file tests typos detection capabilities


def test_spelling_errors():
    """Function containing various spelling errors from the commit."""

    # Error: dimention -> dimension
    check_last_dimention = "must be contiguous at last dimention"

    # Error: implememntation -> implementation
    shared_expert_implememntation = "shared expert implememntation for int8"

    # Error: luckly -> luckily
    comment_luckly = "oops! 4x4 spills but luckly we use 4x2"

    # Error: hanel -> handle
    doesnt_hanel_nan = "this doesn't hanel NaN."

    # Error: build-in -> built-in
    flag_buildin = "Flag enables build-in KV-connector dependency libs"

    # Error: suported -> supported
    not_yet_suported = "are not yet suported"

    # Error: yeild -> yield
    will_yeild_same = "will yeild the same values"

    # Error: referrence -> reference
    triton_referrence = "triton referrence"

    # Error: assums -> assumes
    grouped_topk_assums = "Since grouped_topk assums top-2"

    # Error: parallell -> parallel
    tensor_parallell_size = "Default tensor parallell size to use"

    # Error: re-use -> reuse (hyphenation)
    reuse_kv_cache = "re-use the kv cache, full attention"

    # Error: Choosen -> Chosen
    choosen_to_be = "Choosen to be the same as sglang"

    # Error: splitted -> split
    requests_splitted = "requests might be splitted into multiple"

    # Error: becuase -> because
    all_cases_becuase = "all cases becuase they only support"

    # Error: Reponses -> Responses
    reponses_api = "Reponses API supports simple text inputs"

    return {
        "dimention": check_last_dimention,
        "implememntation": shared_expert_implememntation,
        "luckly": comment_luckly,
        "hanel": doesnt_hanel_nan,
        "buildin": flag_buildin,
        "suported": not_yet_suported,
        "yeild": will_yeild_same,
        "referrence": triton_referrence,
        "assums": grouped_topk_assums,
        "parallell": tensor_parallell_size,
        "reuse": reuse_kv_cache,
        "choosen": choosen_to_be,
        "splitted": requests_splitted,
        "becuase": all_cases_becuase,
        "reponses": reponses_api
    }


class TestSpellingClass:
    """Class with spelling errors for testing."""

    def __init__(self):
        self.dimention_error = "last dimention"
        self.implememntation_note = "expert implememntation"
        self.luckly_comment = "luckly we use 4x2"

    def process_hanel_error(self):
        """Method that doesn't hanel NaN properly."""
        return "doesn't hanel NaN"

    def check_suported_models(self):
        """Check if models are suported."""
        return "not yet suported"

    def yeild_results(self):
        """Method that will yeild same values."""
        return "will yeild the same"


# Additional spelling errors from the commit
COMMON_ERRORS = {
    "referrence": "triton referrence implementation",
    "assums": "function assums top-2 selection",
    "parallell": "tensor parallell processing",
    "choosen": "algorithm choosen for optimization",
    "splitted": "data splitted into chunks",
    "becuase": "used becuase of performance",
    "reponses": "API reponses handling"
}
