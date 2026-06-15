# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Return your answer as a JSON object with exactly these keys:
{"label": "<one of interview, solo, panel, narrative>", "reasoning": "<one or two sentences>"}
Return only the JSON, with no extra text.

```

---

**Edge cases to handle in the prompt:**

```
- Empty labeled_examples: the prompt should still work as a zero-shot
  classifier. Skip the examples section entirely (don't emit an empty
  "Examples:" header), keep the task instruction and the four label
  definitions, then present the episode to classify. The label
  definitions carry the classification signal when no examples exist.

- Very short or empty description: still build the prompt normally and
  let the model pick the best-fit label from the four. Don't pad or
  fabricate description text. ("unknown" is reserved for parse/API
  failures, handled in classify_episode Step 5 — not a label the prompt
  asks for.)

- Extremely long labeled_examples list: optionally cap the number of
  examples included (e.g., first N) so the prompt doesn't blow past the
  token limit. Note the cap if you add one.

```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The prompt requests a JSON object, so parse with json.loads():

  1. Strip whitespace from response.choices[0].message.content.
  2. Models sometimes wrap JSON in ```json ... ``` fences or add stray
     text. Defensively slice from the first "{" to the last "}" before
     parsing (text[text.find("{"): text.rfind("}") + 1]).
  3. Call json.loads() on that substring inside a try/except (see Step 5).
  4. Read data["label"] and data["reasoning"]. Normalize the label with
     .strip().lower() so casing/whitespace differences don't fail
     validation in Step 4.
  5. If a key is missing, treat it as an unparseable response and fall
     through to the error path (label = "unknown").
```

---

**Step 4 — Validate the label:**

```
After normalizing (.strip().lower()), check membership:

  if label not in VALID_LABELS:
      label = "unknown"

This catches hallucinated labels, near-misses ("interviews", "monologue"),
empty strings, and anything outside the four allowed values. Keep the
reasoning text even when the label is invalid — it's useful for debugging
why the model went off-taxonomy. VALID_LABELS stays the single source of
truth, so adding a label later only requires updating that constant.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the API call and parsing in try/except so a single failure never
crashes the 20-call evaluation loop. Things that can go wrong:

  - Network / API error (timeout, rate limit, auth) from
    _client.chat.completions.create()
  - json.loads() raises ValueError on non-JSON or truncated output
  - KeyError if "label" or "reasoning" is missing from the parsed dict

On any failure, return a valid dict so the caller can keep going:

  {"label": "unknown", "reasoning": f"error: {e}"}

This guarantees the function ALWAYS returns the documented shape
(keys "label" and "reasoning", label is a VALID_LABELS member or
"unknown"). Optionally print/log the error so failures are visible
during evaluation rather than silently counted as "unknown".
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: [Five Writers on What It Means to Write for the Internet Now]
Raw response text: [panel]
Reasoning: [The episode features five writers discussing how the internet has changed what it means to publish writing, with a focus on their differing opinions and a frank conversation, indicating a panel format with multiple guests and roughly equal speaking time.]
```

**How did you parse the label out of the response?**

```
The prompt asks for a JSON object, so I parsed with json.loads():

  1. Took response.choices[0].message.content and stripped whitespace.
  2. Sliced from the first "{" to the last "}" (text[text.find("{"):
     text.rfind("}") + 1]) to drop any ```json fences or stray text the
     model added around the JSON.
  3. json.loads() on that substring, inside a try/except.
  4. Pulled data["label"] and data["reasoning"], then normalized the
     label with .strip().lower() before the VALID_LABELS check.

So the key string operations were: strip, find/rfind to isolate the
JSON, json.loads to parse, and .strip().lower() to normalize the label.
```

**Did any episodes return `"unknown"`? If so, why?**

```
[no]
```

**One thing about the output format that surprised you:**

```
[nothing]
```
