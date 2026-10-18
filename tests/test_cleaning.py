from virallens.common.cleaning import Line, clean_document

ECF = "Case 2:25-cv-01920-BRM-CLW Document 9 Filed 04/11/25 Page {n} of 51 PageID: {id}"


def _body_page(n: int, body: str) -> str:
    return f" {n} \n{body}\n{ECF.format(n=n, id=65 + n)}"


BODIES = [
    "The Fund is a series of Upright Trust.",
    "Chiueh founded Upright in the 1990s.",
    "The Concentration Policy caps one industry at 25%.",
    "The 2021 Order found violations of that policy.",
    "Defendants continued to invest in Company A.",
]


def test_recurring_page_numbers_and_ecf_stamps_are_removed():
    pages = [_body_page(n, body) for n, body in enumerate(BODIES, start=1)]
    assert clean_document(pages) == [Line(n, body) for n, body in enumerate(BODIES, start=1)]


def test_underline_rule_does_not_shield_running_head():
    # Four body lines keep the running head (4th line) outside the bottom edge window too, as on
    # the real 25+ line pages.
    rules = "_________________\n_________________"
    bodies = [[f"{body} Sentence {k}." for k in "ABCD"] for body in BODIES]
    pages = [
        f"{rules}\n{n} Cite as: 601 U. S. ____ (2024)\nOpinion of the Court\n" + "\n".join(lines)
        for n, lines in enumerate(bodies, start=1)
    ]
    expected = [Line(n, text) for n, lines in enumerate(bodies, start=1) for text in lines]
    assert clean_document(pages) == expected


def test_edge_line_seen_on_too_few_pages_is_kept():
    pages = ["SUMMARY\nbody one", "body two", "body three", "body four"]
    assert Line(1, "SUMMARY") in clean_document(pages)


def test_edge_line_below_the_page_share_is_kept():
    # Recurs on 3 of 20 pages: enough pages, but under a quarter of the document.
    pages = [
        f"{'SECURITIES AND EXCHANGE COMMISSION' if n < 3 else 'Body'}\nText {n}" for n in range(20)
    ]
    assert (
        sum(line.text == "SECURITIES AND EXCHANGE COMMISSION" for line in clean_document(pages))
        == 3
    )


def test_furniture_pattern_in_mid_page_is_kept():
    body = "\n".join(
        f"{word} sentence of the body." for word in ["First", "Second", "Third", "Fourth"]
    )
    pages = [f"{n}\n{body}\n25\n{body}\n{n}" for n in range(1, 6)]
    assert [line.text for line in clean_document(pages)].count("25") == 5


def test_table_of_contents_page_is_dropped():
    toc = "TABLE OF CONTENTS\nOpinions below ........ 1\nJurisdiction ......... 1\nStatement .... 2"
    assert clean_document(["Cover", toc, "Body"]) == [Line(1, "Cover"), Line(3, "Body")]


def test_body_page_with_one_leader_line_is_kept():
    body = "\n".join(f"Sentence {k} of the body." for k in range(9))
    assert len(clean_document([f"{body}\nDated ........ April 11, 2025"])) == 10


def test_syllable_break_is_joined_without_hyphen():
    assert clean_document(["the “Risk Factors” sec-\ntion of an SEC Form"]) == [
        Line(1, "the “Risk Factors” section of an SEC Form")
    ]


def test_capitalised_syllable_break_is_joined():
    assert clean_document(["Siracusano, 563 U.S. 27, 45 (2011). Accord-\ningly, disclosure"]) == [
        Line(1, "Siracusano, 563 U.S. 27, 45 (2011). Accordingly, disclosure")
    ]


def test_hyphen_kept_when_document_spells_the_compound_hyphenated():
    raw = ["Each risk-factor statement matters.\nAn incomplete risk-\nfactor statement"]
    assert clean_document(raw)[-1] == Line(1, "An incomplete risk-factor statement")


def test_citation_hyphen_before_a_digit_is_kept():
    assert clean_document(["under Rule 10b-\n5(b), a specialist"]) == [
        Line(1, "under Rule 10b-5(b), a specialist")
    ]


def test_break_with_stray_space_before_hyphen_is_joined():
    assert clean_document(["information that a reason -\nable investor"]) == [
        Line(1, "information that a reasonable investor")
    ]


def test_word_continued_on_next_page_keeps_first_page_number():
    assert clean_document(["the Court has held that disclo-", "sure is required"]) == [
        Line(1, "the Court has held that disclosure is required")
    ]


def test_space_before_intra_word_hyphen_is_removed():
    assert clean_document(["a Form 10 -K and the warned -of risk"]) == [
        Line(1, "a Form 10-K and the warned-of risk")
    ]


def test_word_spelled_both_ways_is_joined_solid():
    raw = ["nonpublic data and non-public data.\nThe non-\npublic data"]
    assert clean_document(raw)[-1] == Line(1, "The nonpublic data")


def test_hyphen_kept_when_compound_is_spelled_with_stray_space_elsewhere():
    raw = ["the virtual -certainty standard. Reject the virtual -\ncertainty rule."]
    assert clean_document(raw) == [
        Line(1, "the virtual-certainty standard. Reject the virtual-certainty rule.")
    ]


def test_hyphen_kept_inside_a_multi_part_compound():
    assert clean_document(["a cut-and-\ndry rule"]) == [Line(1, "a cut-and-dry rule")]


def test_en_dash_inside_a_term_becomes_a_hyphen():
    assert clean_document(["Rule 10b–5 and Regulation S–K, under Rule 10b–\n5(b)"]) == [
        Line(1, "Rule 10b-5 and Regulation S-K, under Rule 10b-5(b)")
    ]
