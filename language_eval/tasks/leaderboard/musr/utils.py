import ast


def doc_to_choice(doc):
    """
    Convert a doc to a choice.
    """
    return ast.literal_eval(doc["choices"])


DOC_TO_TEXT = "{narrative}\n\n{question}\n\n{choices}\nAnswer:"


def doc_to_text(doc):
    """
    Convert a doc to text.
    """
    choices = ""
    for i, choice in enumerate(ast.literal_eval(doc["choices"])):
        choices += f"{i + 1} - {choice}\n"

    text = DOC_TO_TEXT.format(
        narrative=doc["narrative"], question=doc["question"], choices=choices
    )

    return text


DOC_TO_TEXT_CHAT = (
    "{narrative}\n\n{question}\n\n{choices}"
    'Your response should end with "The best answer is [number]" '
    "where [number] is the number of the correct choice.\nAnswer:"
)


def doc_to_text_chat(doc):
    """Convert a doc to text for the generate_until (chat) variant."""
    choices = ""
    for i, choice in enumerate(ast.literal_eval(doc["choices"])):
        choices += f"{i + 1} - {choice}\n"

    return DOC_TO_TEXT_CHAT.format(
        narrative=doc["narrative"], question=doc["question"], choices=choices
    )


def doc_to_target_chat(doc):
    """Return the 1-indexed answer number as a string."""
    return str(doc["answer_index"] + 1)
