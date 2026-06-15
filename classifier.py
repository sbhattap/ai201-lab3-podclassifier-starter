import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    # Cap the number of examples so the prompt doesn't blow past the token limit.
    MAX_EXAMPLES = 20

    task_instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    output_format = (
        'Return your answer as a JSON object with exactly these keys:\n'
        '{"label": "<one of interview, solo, panel, narrative>", '
        '"reasoning": "<one or two sentences>"}\n'
        "Return only the JSON, with no extra text."
    )

    parts = [task_instruction]

    # Examples section — skipped entirely when there are none (zero-shot).
    if labeled_examples:
        example_blocks = []
        for ex in labeled_examples[:MAX_EXAMPLES]:
            example_blocks.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Label: {ex['label']}"
            )
        parts.append("Examples:\n\n" + "\n\n---\n\n".join(example_blocks))

    # The new episode to classify, in the same format but with the label withheld.
    parts.append(
        "Title: (the episode below)\n"
        f"Description: {description}\n"
        "Label: ?"
    )

    parts.append(
        "Classify the episode above. Return your answer in the format below:\n\n"
        + output_format
    )

    return "\n\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    # Step 1 — Build the prompt.
    prompt = build_few_shot_prompt(labeled_examples, description)

    try:
        # Step 2 — Send to the LLM.
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()

        # Step 3 — Parse the response. Defensively slice to the JSON object in
        # case the model wraps it in ```json fences or adds stray text.
        json_text = text[text.find("{"): text.rfind("}") + 1]
        data = json.loads(json_text)

        label = data["label"].strip().lower()
        reasoning = data["reasoning"]

        # Step 4 — Validate the label.
        if label not in VALID_LABELS:
            label = "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        # Step 5 — Handle errors gracefully so one failure can't crash the
        # evaluation loop (API errors, bad/truncated JSON, missing keys).
        print(f"classify_episode error: {e}")
        return {"label": "unknown", "reasoning": f"error: {e}"}
