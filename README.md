# Group project - an end-to-end modelling report

Working in your team, you will take a question you can defend, build the smallest model
that answers it honestly, and write it up so that a reader who was not in the room can
tell whether to believe you.

## Task

The point of this project is not the highest possible score; a careful, honestly
evaluated simple model beats a leaky, over-tuned ensemble every time, and the report is
where you show your reasoning stands up. Agree your question and dataset early, and write
the question down before you touch the data - a team that decides the question after
seeing the results has already made the mistake the marking looks for.

Build a baseline first and keep its number in the final report next to your model's.
Every fitted transformation - scaling, encoding, imputation - has to sit inside your
train/test split, never fitted on the whole dataset beforehand; this is the single most
common way an otherwise sound project quietly leaks, and everyone on the team should be
able to explain why it matters.

To get started:

- Agree the question and dataset as a team in the first week, not the third.
- Build and report a baseline before tuning anything more elaborate.
- Fit every transformation inside the split, using a pipeline rather than doing it by hand.
- Split once, early, and do not return to the test set until the report is written.
- Draft `REPORT.md` early and keep it updated as the model develops.

## What to submit

This is group work: one submission per team, from your team's own repository. Push
`REPORT.md`, `src/pipeline.py` (or the R equivalent), `requirements.txt` (or `renv.lock`)
and `data/README.md` describing where the data came from, to `main`. Do not commit the raw
data itself.

## Grading

- Framing and data handling: a real question, a metric that matches the decision, and
  missingness and provenance handled deliberately.
- Method and evaluation: a defensible split, honest tuning, and comparison against a
  baseline with an interval.
- Report and model card: legible to a non-specialist, with limitations stated in specifics
  rather than in general terms.
- Reproducibility: runs from a clean checkout with a fixed seed and pinned dependencies;
  each member's mark carries an individual adjustment based on their contribution.
