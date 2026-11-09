"""The lenient rule of qa_transfer/lenient.py (version 2): what counts as a letter, an option's text, or nothing."""

from eval_external.qa_transfer import lenient as L

OPTS = ["A. chair", "B. dining table", "C. table", "D. back-left"]


def test_letters_written_other_ways():
    for pred, want in [("B", "B"), ("(B)", "B"), ("[C]", "C"), ("B.", "B"), ("B) dining table", "B"),
                       ("(B) dining table", "B"), ("Answer: D", "D"), ("The answer is (C).", "C"),
                       ("<answer>B. dining table</answer>", "B"), ("<think>A or B</think>(D)", "D"),
                       ("<C. table>", "C"), ("the correct option is B", "B")]:
        assert L.lenient_letter(pred, OPTS) == (want, "letter"), pred


def test_option_text():
    assert L.lenient_letter("dining table", OPTS) == ("B", "text")
    assert L.lenient_letter("A dining table.", OPTS) == ("B", "text")      # article, not option A
    assert L.lenient_letter("table", OPTS) == ("C", "text")                 # equal beats contained
    assert L.lenient_letter("the back-left one", OPTS) == ("D", "text")
    assert L.lenient_letter("it is the dining table", OPTS) == ("B", "text")  # longer option contains shorter
    assert L.lenient_letter("the answer is a table", OPTS) == ("C", "text")  # lowercase 'a' is not option A


def test_unparseable():
    assert L.lenient_letter("", OPTS) == (None, "unparseable")
    assert L.lenient_letter("I cannot tell from these images.", OPTS) == (None, "unparseable")
    assert L.lenient_letter("E", OPTS) == (None, "unparseable")             # not an option of this item
    assert L.lenient_letter("chair or table", OPTS) == (None, "unparseable")
    assert L.lenient_letter("(B) or (C)", OPTS) == (None, "unparseable")     # two labels named
    assert L.lenient_letter("It is not the chair", OPTS) == (None, "unparseable")  # negation: no containment
    cut = "Let me check each.\nOption D: back-left, impossible because\nOption C: table, which"
    assert L.lenient_letter(cut, OPTS) == (None, "unparseable")             # options listed, no answer given


def test_labels_not_positions():
    opts = ["A. left", "B. right", "E. front", "F. back", "C. up", "D. down"]
    assert L.option_labels(opts) == ["A", "B", "E", "F", "C", "D"]
    assert L.lenient_letter("front", opts) == ("E", "text")
    assert L.lenient_letter("D", opts) == ("D", "letter")


def test_numbers_and_mra():
    assert L.first_number("about 2.5 meters") == 2.5
    assert L.first_number("three") == 3.0
    assert L.first_number("no idea") is None
    assert L.mra_vlmeval(2.0, 2.0) == 1.0 and L.mra_vsti(2.0, 2.0) == 1.0
    assert L.mra_vlmeval(4.0, 2.0) == 0.0
    assert abs(L.mra_vlmeval(2.2, 2.0) - 0.9) < 1e-9   # error 0.1 passes thresholds 0.5..0.9 (9 of 10)


def test_official_parse_stands_or_is_overridden():
    it = {"official_parsed": True, "official": 0.0, "official_letter": "A", "mcq": True, "pred": "A dining table",
          "options": OPTS, "answer": "B"}
    assert L.rescore_item(it) == {"lenient": 0.0, "how": "official"}        # 'A' stands alone: backend's call
    opts = ["A. Directly right", "B. Directly left", "C. Diagonally forward", "D. Diagonally back"]
    it = {"official_parsed": True, "official": 0.0, "official_letter": "D", "mcq": True,
          "pred": "<answer>Directly left</answer>", "options": opts, "answer": "B"}
    assert L.rescore_item(it) == {"lenient": 1.0, "how": "text_over_official", "letter": "B"}  # D from a word
    it = dict(it, official_parsed=False, official_letter=None, pred="dining table", options=OPTS)
    assert L.rescore_item(it)["lenient"] == 1.0
