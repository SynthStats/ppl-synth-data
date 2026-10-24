You are evaluating a data analysis prompt for information leakage.

The prompt should describe a dataset and research question using only
observable data properties and domain context: it should NOT reveal
which statistical model or distribution family to use.

Given ONLY the prompt below (no other context), can you identify the
specific probability distribution family or model class intended?

Rate the leakage on this scale:
- CLEAN: Prompt describes data properties only. Model choice requires genuine reasoning.
- MILD: Prompt contains subtle hints (e.g., "exponential growth" suggests exponential family)
  but a reasonable statistician would consider multiple model families.
- LEAKED: Prompt makes the model family obvious (names distributions, uses canonical
  parameters, gives explicit modeling directives).

Output valid JSON only. No prose. No markdown fences.

Schema:
{"rating": "CLEAN|MILD|LEAKED", "evidence": "brief explanation", "recovered_family": "family name or null"}
