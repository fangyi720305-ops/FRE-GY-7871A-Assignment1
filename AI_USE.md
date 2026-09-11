
# AI use disclosure

**Tools used:** OpenAI ChatGPT and Codex.

**What I used them for:** I used AI to help organize the analysis notebook, draft Python code for dictionary scoring, quarterly aggregation, market-variable construction, regressions, tables, and figures, and debug errors involving two-way clustered standard errors and pandas merges. I also used it to help interpret the statistical results and edit the report for clarity and concision.

**What I wrote myself:** I ran the four starter scripts, executed and reviewed every notebook cell, checked intermediate sample counts and outputs, made the final methodological choices, and verified that the reported tables, figure, and conclusions matched the executed results. I also reviewed and adapted all AI-assisted code before including it in the submission.

**Anything the model got wrong that I had to correct:** An initial two-way clustering implementation passed string/object cluster labels to `statsmodels`, causing a NumPy data-type error; I corrected it by converting both clustering variables to integer codes. An initial merge incorrectly used `one_to_one` as the merge method instead of as the validation rule. I also identified and corrected implausible market-cap controls caused by XBRL share-count units and stock-split timing before running the final regressions.
