# Independent adverse-media annotation rubric

## Unit of annotation

Classify the named `entity_name`, using only the supplied article. Do not classify the article's overall topic or the conduct of another entity.

## Labels

### 1: good_guy

Use label 1 when the named entity is not described as participating in adverse conduct. This includes:

- Victims and targets of crime.
- Police officers, investigators, prosecutors, judges, witnesses, whistleblowers, rescuers, and others clearly acting appropriately.
- Authors, photographers, publishers, and quoted experts who merely report or discuss the conduct.
- Organizations or people mentioned incidentally or only connected to the victim.
- Employers, relatives, schools, teams, and other associates when the article does not attribute participation, enabling, facilitation, or responsibility to them.
- Completely unrelated entities with the same or a similar name.

### 2: bad_guy

Use label 2 when the article attributes to the named entity direct, indirect, verified, suspected, current, or past participation in criminal, unethical, adverse, detrimental, or sanctioned conduct. This includes:

- Convicted, sentenced, imprisoned, fined, sanctioned, or otherwise verified participants.
- Accused, charged, arrested, wanted, suspected, investigated, or formally alleged participants, regardless of denial or presumption of innocence.
- People or organizations that knowingly or unknowingly enabled, facilitated, collaborated in, concealed, financed, or benefited from the conduct when the article attributes that connection to them.
- Past misconduct and current associations with involved entities when participation or facilitation is attributed.

## Decision rules

1. Resolve the named entity first. A criminal appearing in the same article does not make the target entity label 2.
2. Use only claims present in the article. Do not add outside knowledge.
3. A denial does not erase an accusation or investigation; label 2 when both appear.
4. Acquittal, exoneration, mistaken identity, or an explicit finding of no involvement is label 1 when the article no longer attributes suspected participation to the entity.
5. Mere employment, family, geographic, contractual, or organizational association is label 1 unless participation, enabling, responsibility, or facilitation is attributed.
6. If the article text is insufficient to identify the entity's role, choose the most defensible label, set confidence below 0.60, and explain the ambiguity. Do not infer from URLs, filenames, monitoring IDs, source labels, or relevancy scores.
7. If the full reference name is not a literal match in the article, do not assume the entity is unrelated. The article may use a surname, acronym, alias, translated name, or variant spelling. Mark `reference_resolution` as `needs_entity_resolution` and require review. Resolve the alias from article text only; otherwise apply rule 6.

## Annotation record

Each prediction must be written separately from the blind article corpus:

```json
{"article_id":"...","label":2,"label_name":"bad_guy","confidence":0.95,"rationale":"The entity is described as charged with fraud.","evidence":["Authorities charged ... with fraud"],"annotator":"github-copilot","status":"ai_annotated"}
```

AI annotations are candidate labels, not ground truth. Promote them to `human_reviewed` only after independent review, with priority given to low-confidence cases and a random sample of high-confidence cases.