import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def _process_doc_chat(doc):
    """Process a doc for the generate_until (chat) variant."""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    query = preprocess(doc["activity_label"] + ": " + ctx)
    choices = [preprocess(ending) for ending in doc["endings"]]
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{letters[i]}. {c}" for i, c in enumerate(choices)
    )
    out_doc = {
        "query": f"Given the following context and four candidate endings (A, B, C and D), choose the most plausible ending.\nContext: {query}\n{choices_text}\nYour response should end with \"The best answer is [letter]\" where [letter] is one of A, B, C or D.",
        "choices": choices,
        "gold": int(doc["label"]),
        "label": letters[int(doc["label"])],
    }
    return out_doc


def process_docs_chat(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc_chat)


def doc_to_text_chat(doc):
    return doc["query"]


def _filter_by_source(dataset: datasets.Dataset, source: str) -> datasets.Dataset:
    return dataset.filter(lambda x: x["source_id"].startswith(source))


def _filter_by_activity_label(dataset: datasets.Dataset, label: str) -> datasets.Dataset:
    return dataset.filter(lambda x: x["activity_label"] == label)


ACTIVITYNET_CATEGORIES = {
    "sports": {
        "Running a marathon", "Clean and jerk", "High jump",
        "Layup drill in basketball", "Futsal", "Cheerleading", "Rope skipping",
        "Using parallel bars", "Playing beach volleyball", "Playing water polo",
        "Playing badminton", "Elliptical trainer", "Doing crunches",
        "Playing lacrosse", "Baton twirling", "Swimming", "Powerbocking",
        "Rollerblading", "Tennis serve with ball bouncing", "Shuffleboard",
        "Plataform diving", "Doing a powerbomb", "Playing polo", "Beach soccer",
        "Discus throw", "Shot put", "Slacklining", "Triple jump", "Croquet",
        "Using the pommel horse", "Using the rowing machine", "BMX",
        "Skateboarding", "Ping-pong", "Dodgeball", "Hurling",
        "Using uneven bars", "Playing ice hockey", "Hula hoop", "Tumbling",
        "Long jump", "Table soccer", "Hopscotch", "Pole vault", "Cricket",
        "Javelin throw", "Playing squash", "Volleyball", "Sumo",
        "Playing kickball", "Breakdancing", "Zumba", "Spinning",
        "Rock climbing", "Longboarding", "Paintball", "Archery", "Snatch",
    },
    "music": {
        "Playing violin", "Playing harmonica", "Playing guitarra",
        "Playing drums", "Playing bagpipes", "Playing saxophone",
        "Playing piano", "Playing congas", "Playing accordion", "Drum corps",
    },
    "water": {
        "Ice fishing", "Scuba diving", "Surfing", "Waterskiing",
        "Snow tubing", "Canoeing", "Wakeboarding", "River tubing",
        "Kayaking", "Windsurfing", "Sailing",
    },
    "home": {
        "Cutting the grass", "Shoveling snow", "Cleaning windows",
        "Ironing clothes", "Roof shingle removal", "Painting furniture",
        "Laying tile", "Blowing leaves", "Hanging wallpaper", "Raking leaves",
        "Installing carpet", "Cleaning shoes", "Polishing shoes",
        "Mooping floor", "Polishing forniture", "Vacuuming floor",
        "Mowing the lawn", "Chopping wood", "Cleaning sink",
        "Hand washing clothes", "Assembling bicycle", "Welding", "Plastering",
        "Fixing bicycle", "Spread mulch", "Removing ice from car",
        "Trimming branches or hedges", "Fixing the roof",
        "Decorating the Christmas tree", "Wrapping presents",
        "Carving jack-o-lanterns", "Hitting a pinata", "Starting a campfire",
    },
    "personal_care": {
        "Washing face", "Gargling mouthwash", "Applying sunscreen",
        "Getting a piercing", "Putting in contact lenses", "Blow-drying hair",
        "Removing curlers", "Shaving", "Getting a haircut", "Putting on makeup",
        "Braiding hair", "Shaving legs", "Washing hands", "Brushing teeth",
        "Putting on shoes", "Getting a tattoo",
    },
    "food": {
        "Having an ice cream", "Baking cookies", "Making a lemonade",
        "Making a cake", "Making a sandwich", "Preparing pasta",
        "Preparing salad", "Mixing drinks", "Drinking beer",
        "Smoking hookah", "Drinking coffee",
    },
    "combat": {
        "Arm wrestling", "Capoeira", "Doing karate", "Doing fencing",
        "Bullfighting", "Doing kickboxing",
    },
}

# Reverse mapping: activity_label -> category
_ACTIVITY_TO_CATEGORY = {}
for _cat, _labels in ACTIVITYNET_CATEGORIES.items():
    for _label in _labels:
        _ACTIVITY_TO_CATEGORY[_label] = _cat


def _filter_activitynet_category(dataset: datasets.Dataset, category: str) -> datasets.Dataset:
    labels = ACTIVITYNET_CATEGORIES[category]
    return dataset.filter(
        lambda x: x["source_id"].startswith("activitynet")
        and x["activity_label"] in labels
    )


def _filter_activitynet_other(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(
        lambda x: x["source_id"].startswith("activitynet")
        and x["activity_label"] not in _ACTIVITY_TO_CATEGORY
    )


def process_docs_activitynet(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_source(dataset, "activitynet"))


def process_docs_activitynet_sports(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "sports"))


def process_docs_activitynet_music(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "music"))


def process_docs_activitynet_water(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "water"))


def process_docs_activitynet_home(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "home"))


def process_docs_activitynet_personal_care(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "personal_care"))


def process_docs_activitynet_food(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "food"))


def process_docs_activitynet_combat(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_category(dataset, "combat"))


def process_docs_activitynet_other(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_activitynet_other(dataset))


def process_docs_wikihow_personal_care(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Personal Care and Style"))


def process_docs_wikihow_family_life(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Family Life"))


def process_docs_wikihow_food(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Food and Entertaining"))


def process_docs_wikihow_computers(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Computers and Electronics"))


def process_docs_wikihow_health(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Health"))


def process_docs_wikihow_home_garden(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Home and Garden"))


def process_docs_wikihow_finance(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Finance and Business"))


def process_docs_wikihow_pets(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Pets and Animals"))


def process_docs_wikihow_education(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Education and Communications"))


def process_docs_wikihow_youth(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Youth"))


def process_docs_wikihow_sports(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Sports and Fitness"))


def process_docs_wikihow_relationships(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Relationships"))


def process_docs_wikihow_work(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Work World"))


def process_docs_wikihow_cars(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Cars & Other Vehicles"))


def process_docs_wikihow_travel(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Travel"))


def process_docs_wikihow_holidays(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Holidays and Traditions"))


def process_docs_wikihow_philosophy(dataset: datasets.Dataset) -> datasets.Dataset:
    return process_docs(_filter_by_activity_label(dataset, "Philosophy and Religion"))


def process_docs_wikihow_other(dataset: datasets.Dataset) -> datasets.Dataset:
    """Uncategorized + Home,Categories (9 total)"""
    return process_docs(dataset.filter(
        lambda x: x["source_id"].startswith("wikihow")
        and x["activity_label"] in ("Uncategorized", "Home,Categories")
    ))


# --- Chat (generate_until) variants ---
# These apply the same filters but use _process_doc_chat instead of process_docs

def _to_chat(filter_fn, dataset: datasets.Dataset) -> datasets.Dataset:
    """Apply a filter function then the chat doc processor."""
    filtered = filter_fn(dataset)
    return filtered.map(_process_doc_chat)


def process_docs_chat_activitynet_sports(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "sports"), ds)

def process_docs_chat_activitynet_music(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "music"), ds)

def process_docs_chat_activitynet_water(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "water"), ds)

def process_docs_chat_activitynet_home(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "home"), ds)

def process_docs_chat_activitynet_personal_care(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "personal_care"), ds)

def process_docs_chat_activitynet_food(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "food"), ds)

def process_docs_chat_activitynet_combat(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_activitynet_category(d, "combat"), ds)

def process_docs_chat_activitynet_other(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(_filter_activitynet_other, ds)

def process_docs_chat_wikihow_personal_care(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Personal Care and Style"), ds)

def process_docs_chat_wikihow_family_life(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Family Life"), ds)

def process_docs_chat_wikihow_food(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Food and Entertaining"), ds)

def process_docs_chat_wikihow_computers(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Computers and Electronics"), ds)

def process_docs_chat_wikihow_health(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Health"), ds)

def process_docs_chat_wikihow_home_garden(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Home and Garden"), ds)

def process_docs_chat_wikihow_finance(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Finance and Business"), ds)

def process_docs_chat_wikihow_pets(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Pets and Animals"), ds)

def process_docs_chat_wikihow_education(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Education and Communications"), ds)

def process_docs_chat_wikihow_youth(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Youth"), ds)

def process_docs_chat_wikihow_sports(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Sports and Fitness"), ds)

def process_docs_chat_wikihow_relationships(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Relationships"), ds)

def process_docs_chat_wikihow_work(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Work World"), ds)

def process_docs_chat_wikihow_cars(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Cars & Other Vehicles"), ds)

def process_docs_chat_wikihow_travel(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Travel"), ds)

def process_docs_chat_wikihow_holidays(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Holidays and Traditions"), ds)

def process_docs_chat_wikihow_philosophy(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(lambda d: _filter_by_activity_label(d, "Philosophy and Religion"), ds)

def process_docs_chat_wikihow_other(ds: datasets.Dataset) -> datasets.Dataset:
    return _to_chat(
        lambda d: d.filter(
            lambda x: x["source_id"].startswith("wikihow")
            and x["activity_label"] in ("Uncategorized", "Home,Categories")
        ),
        ds,
    )
