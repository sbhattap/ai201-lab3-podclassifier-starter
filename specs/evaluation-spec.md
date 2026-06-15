# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[accuracy = no. of correct predictions / total number of episodes]
```

---

**Step-by-step logic:**

```
1. Take the total number of episodes as the length of `ground_truth`
   (predictions and ground_truth are the same length, paired by index).
2. If the total is 0, return 0.0 immediately (avoid dividing by zero).
3. Initialize a counter `correct = 0`.
4. Walk through the two lists in lockstep (zip predictions with ground_truth).
5. For each (predicted, truth) pair, if predicted == truth, increment `correct`.
6. After the loop, return correct / total as a float between 0.0 and 1.0.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0. With no episodes there are no correct predictions and the total
is 0, so the fraction is undefined — dividing would raise ZeroDivisionError.
Returning 0.0 keeps the type contract (a float in [0.0, 1.0]) and is safe for
the report formatter, which prints the value directly.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[correct predictions: 2 ("interview" and "solo")
total episodes: 4
accuracy = 2 / 4 = 0.5]
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[An "interview" episode counts as correctly classified if its ground-truth label is
 "interview" and the predicted label is also "interview". The same goes for the other labels. Correct for a given class just mean the no. of correctly classified episodes for that class.]
```

---

**What does "total" mean for a given class?**

```
[Total for a given class means the number of episodes in the ground truth that have that label, regardless of the prediction. For example, if there are 5 "interview" episodes in the ground truth, then total for "interview" is 5.]
```

---

**Step-by-step logic:**

```
1. Initialize a stats dict with one entry per label in VALID_LABELS, each
   starting at {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over the (predicted, truth) pairs by zipping predictions with
   ground_truth.
3. For each pair (predicted, truth):
     a. The truth label decides which class this episode belongs to, so
        increment stats[truth]["total"] by 1.
     b. If predicted == truth, the episode was classified correctly, so also
        increment stats[truth]["correct"] by 1.
   (A truth label not in VALID_LABELS shouldn't occur; skip it if it does.)
4. After the loop, go through each label in the stats dict and compute its
   accuracy: if total > 0, accuracy = correct / total; otherwise accuracy = 0.0.
5. Return the stats dict, keyed by label, each value holding correct, total,
   and accuracy.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy to 0.0 (matching the docstring in evaluate.py, which says
"accuracy: correct / total (0.0 if total is 0)"). With zero ground-truth
episodes for that class, correct / total is undefined and dividing would
raise ZeroDivisionError. Returning 0.0 avoids the crash and keeps the type
contract (a float). Note this is a convention, not a true measurement — the
class simply has no test data, so its "correct" and "total" stay at 0.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Walk the pairs (bucket by the ground-truth label):
  1. pred=interview, truth=interview  -> interview: total+1, correct+1
  2. pred=interview, truth=solo       -> solo:      total+1   (wrong)
  3. pred=solo,      truth=solo       -> solo:      total+1, correct+1
  4. pred=panel,     truth=panel      -> panel:     total+1, correct+1
  5. pred=panel,     truth=narrative  -> narrative: total+1   (wrong)

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0   (1/1)
solo        1        2      0.5   (1/2)
panel       1        1      1.0   (1/1)
narrative   0        1      0.0   (0/1)
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?
   The overall accuracy may be misleading if the dataset is imbalanced like, too many interview episodes and very few narrative episodes. The model could achieve high overall accuracy by just being good at classifying the majority class (interview) while performing poorly on the minority class (narrative). Per-class accuracy reveals how well the model performs on each individual class, highlighting any weaknesses in specific categories that overall accuracy might mask.

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?
   Training labels or my prompt may not be providing enough distinctive features to differentiate between `panel` and `interview` episodes. I may need to review and refine my training examples and prompt to ensure they capture the nuances that distinguish `panel` episodes from `interview` episodes.

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   The evaluation results may improve but it is already at 100% accuracy so it may not really change.
   What if you had 200 test episodes?
   With more test episodes, the evaluation results may become more reliable and stable. The accuracy could decrease if these episodes were tricker to classify.

