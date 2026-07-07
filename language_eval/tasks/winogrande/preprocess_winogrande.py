def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


def doc_to_text_chat(doc):
    sentence = doc["sentence"]
    return (
        f"Given the following sentence with a blank (_), choose which option (A or B) best fills the blank.\n"
        f"Sentence: {sentence}\n"
        f"A. {doc['option1']}\n"
        f"B. {doc['option2']}\n"
        f'Your response should end with "The best answer is [letter]" where [letter] is A or B.'
    )
