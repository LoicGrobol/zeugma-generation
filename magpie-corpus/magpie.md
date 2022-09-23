---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## Load data
<!-- #endregion -->

Download the magpie corpus data (`-C -` option to avoid redownloading it if it's already here and complete)

```python
!curl -C - --create-dirs https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_filtered_split_typebased.jsonl --output local/magpie.jsonl
```

```python
import random
from collections import defaultdict
from typing import Any, NamedTuple

import jsonlines
import rich.progress
import spacy
```

Download the spacy model (if it isn't available already)

```python
!spacy info en_core_web_lg > /dev/null || spacy download en_core_web_lg
```

```python
nlp = spacy.load("en_core_web_lg")
```

```python
idioms = defaultdict(lambda: defaultdict(list))
non_idioms = defaultdict(list)

# We only need POS at this stage
with nlp.select_pipes(enable=["tok2vec", "tagger", "attribute_ruler"]):
    with rich.progress.open("local/magpie.jsonl") as in_stream:
        # Progress bar because it's looooong
        reader = jsonlines.Reader(in_stream)
        for example in reader:
            # We only want idioms that start with a verb
            first_token = nlp(example["idiom"])[0]
            if first_token.pos_ == "VERB":
                verb = first_token.text
                if example["label"] == "i":
                    idioms[verb][example["idiom"]].append(example)
                else:
                    non_idioms[verb].append(example)
        reader.close()
```

That's how many idioms we have (not instances, just idioms)

```python
len(idioms)
```

And this is how many of these we have non-idiomatic usages for

```python
len(set(idioms).intersection(non_idioms))
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## First proof of concept
<!-- #endregion -->

Here's how one sample looks like

```python
sample = idioms["give"]["give someone the creeps"][0]
```

The context always containes 5 sentences, with the idioms located in the third one, and the positions of the word of the idioms given by `offsets`, so this is how to extract uses:

```python
s = sample["offsets"][0][0]
e = sample["offsets"][-1][-1]
print(sample["context"][2][s:e])
```

Let's manipulate the sentence now

```python
sent = nlp(sample["context"][2])
sent
```

Thank the gods for spacy, here's the subtree for the idiom

```python
# Note that this works because we've selected only idioms where a verb is in first position
target_verb = sent.char_span(*sample["offsets"][0])[0]
# print([(t.text, t.head, t.dep_) for t in target_verb.subtree])
subtree = list(target_verb.subtree)
subtree
```

Now we need to split subj and verb (which we factor) from object (that we will coordinate with another object. This is left subtree + head vs right subtree (because we're in English).

```python
sv_span = sent[subtree[0].i:target_verb.i+1]
idiom_obj_span = sent[target_verb.i+1:subtree[-1].i+1]
print("S+V:", sv_span)
print("O:", idiom_obj_span)
```

Now let's find non-idiomatic usages. Here note that the English models in spaCy don't use UD but [ClearNLP labels](https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md)

```python
non_idiom_uses = []
for sample in non_idioms["give"]:
    # We only want to keep samples with the same verb form, thanks English for your inexistant morphology
    form = sample["context"][2][sample["offsets"][0][0]:sample["offsets"][0][-1]]
    if form != target_verb.text:
        continue
    non_idiom_sent = nlp(sample["context"][2])
    non_idiom_target_verb = non_idiom_sent.char_span(*sample["offsets"][0])[0]
    # Ideally, we only want one direct object, that makes it easier to build a coordination
    # And to keep it in the right subtree
    # NOTE: we might want to use stanza instead, since it give UD trees while SpaCy has, well, something
    if len([t for t in non_idiom_target_verb.rights if t.dep_ == "dobj"]) !=1:
        continue
    # We should also control for subjects maybe?
    non_idiom_uses.append(non_idiom_target_verb)
    # print([(t.text, t.head, t.dep_) for t in non_idiom_target_verb.subtree])
    print(list(non_idiom_target_verb.subtree))
```

Let's try with the second one here first. It's a bit weird but serves as a demo. We get the direct object, that we will glue to the idiomatic tree.

```python
non_idiom_verb = non_idiom_uses[1]
non_idiomatic_dobj = next(t for t in non_idiom_verb.rights if t.dep_ == "dobj")
non_idiom_obj_subtree = list(non_idiomatic_dobj.subtree)
non_idiom_obj_span = non_idiom_verb.doc[non_idiom_verb.i+1:non_idiom_obj_subtree[-1].i+1]
non_idiom_obj_span
```

OK, one question is what kind of merges we want to do? Do we merge only identical syntactic frames or do we give some leeway?


In any case, here's how to glue these two together

```python
# There might be a better way to do this
zeugma = " ".join(t.text for span in (sv_span, idiom_obj_span) for t in span) + " and " + " ".join(t.text for t in non_idiom_obj_span)
zeugma
```

<!-- #region tags=[] -->
## Generalizations
<!-- #endregion -->

### Extracting from idioms


Ok, now let's try to generalize all of this. We'll start by using the proof of concept above as a template and iterate on that until we reach satisfactory results.

```python
def extract_from_sample(sample):
    idiom_start = sample["offsets"][0][0]
    idiom_end = sample["offsets"][-1][-1]
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]
    subtree = list(target_verb.subtree)
    sv_span = sent[subtree[0].i:target_verb.i+1]
    obj_span = sent[target_verb.i+1:subtree[-1].i+1]
    return target_verb, sv_span, obj_span

extract_from_sample(idioms["give"]["give someone the creeps"][0])
```

Let's see what that gets us

```python
for sample in idioms["give"]["give someone the creeps"]:
    print(extract_from_sample(sample))
```

OK, that's too many and too complex, let's try to cut these by only extracting the first object (`dobj` for now, we'll see about datives later), like we did for the non-idiomatic case above, and only nomial subjects (`nsubj`, as opposed to e.g. clausal `csubj`). This means that sometimes we'll return `None` if we can't find something to work with in this sample.

```python
def extract_from_sample(sample):
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]

    if (subj := next((t for t in target_verb.lefts if t.dep_ == "nsubj"), None)) is None:
        print(f"No nsubj. sent: {list(zip(sent,[(t.head, t.dep_) for t in sent]))}")
        return None
    subj_subtree = list(subj.subtree)
    # We also want to avoid too long subjects in general so let's ditch everything that has a verb in it
    if any(t for t in subj_subtree if t.pos_ == "VERB"):
        return None
    subj_span = sent[subj_subtree[0].i:subj_subtree[-1].i+1]

    if (obj := next((t for t in target_verb.rights if t.dep_ == "dobj"), None)) is None:
        print(f"No obj. sent: {list(zip(sent,[(t.head, t.dep_) for t in sent]))}")
        return None
    obj_subtree = list(obj.subtree)
    # This departs from the proof-of-concept in that we remove intervening elements between
    # the verb and the object
    obj_span = sent[obj_subtree[0].i:obj_subtree[-1].i+1]
    
    # ATTENTION: the return signature change since we don't have a sv span, but a subj span and a verb
    return subj_span, target_verb, obj_span

for sample in idioms["give"]["give someone the creeps"]:
    extracted = extract_from_sample(sample)
    if extracted is not None:
        print(extracted)
```

That's better, but there's a problem that's obvious with particular idiom but will probably also happen with others: we need the obliques (or dative in this terminology), and indirect objects (pobj here) as well for ditransitive verbs, otherwise this will just sound weird. We also want object predicates (`oprd`, like *wrong* in *put a foot wrong*).So let's try that, but only for cases where there's also a direct object.

```python
def extract_from_sample(sample):
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]

    if (subj := next((t for t in target_verb.lefts if t.dep_ == "nsubj"), None)) is None:
        print(f"No nsubj. sent: {list(zip(sent,[(t.head, t.dep_) for t in sent]))}")
        return None
    subj_subtree = list(subj.subtree)
    # We also want to avoid too long subjects in general so let's ditch everything that has a verb in i
    if any(t for t in subj_subtree if t.pos_ == "VERB"):
        return None
    subj_span = sent[subj_subtree[0].i:subj_subtree[-1].i+1]

    if not any(t for t in target_verb.rights if t.dep_  == "dobj"):
        print(f"No obj. sent: {list(zip(sent,[(t.head, t.dep_) for t in sent]))}")
        return None
    objects = [t for t in target_verb.rights if t.dep_ in ("dative", "dobj", "prep", "oprd")]
    obj_spans = []
    for obj in objects:
        obj_subtree = list(obj.subtree)
        obj_spans.append(sent[obj_subtree[0].i:obj_subtree[-1].i+1])
    
    # ATTENTION: the return signature changes again since we now have multiple object spans
    return subj_span, target_verb, obj_spans

for sample in idioms["give"]["give someone the creeps"]:
    extracted = extract_from_sample(sample)
    if extracted is not None:
        print(extracted)
```

Nearly there for this iteration, but we still have a few issues

- An example that's in a relative clause, and so can't be promoted to main clause (“who both…”): let's remove the verbs that head problematic embedded clauses
- An example that's an adverbial clause, same issue “his smile giving…”
- A missed auxilary in “mice and such like have always given him the creeps…”, we just have to move from a single verb token to a verb span (which will also include the modifier *always* here, but that's optional if we find out later that we don't want it)

```python
def extract_from_sample(sample):
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]
    if target_verb.dep_ in {"acl", "advcl", "relcl", "acomp", "ccomp", "pcomp", "xcomp"}:
        return None
    # Again relying on that nice positional syntax
    verb_span_start = min((t.i for t in target_verb.lefts if t.dep_ == "aux"), default=target_verb.i)
    verb_span = sent[verb_span_start:target_verb.i+1]

    if (subj := next((t for t in target_verb.lefts if t.dep_ == "nsubj"), None)) is None:
        return None
    subj_subtree = list(subj.subtree)
    if any(t for t in subj_subtree if t.pos_ == "VERB"):
        return None
    subj_span = sent[subj_subtree[0].i:subj_subtree[-1].i+1]

    if not any(t for t in target_verb.rights if t.dep_  == "dobj"):
        return None
    objects = [t for t in target_verb.rights if t.dep_ in ("dative", "dobj",  "prep", "oprd")]
    obj_spans = []
    for obj in objects:
        obj_subtree = list(obj.subtree)
        obj_spans.append(sent[obj_subtree[0].i:obj_subtree[-1].i+1])
    
    # ATTENTION: the return signature changes again since we now have a verb span, but we still want to keep
    # the target verb isolated
    return target_verb, subj_span, verb_span, obj_spans

for sample in idioms["give"]["give someone the creeps"]:
    extracted = extract_from_sample(sample)
    if extracted is not None:
        print(extracted)
```

Let's extend this to other *give* idioms:

```python
for idiom, samples in idioms["give"].items():
    print(f"idiom: '{idiom}'")
    for sample in samples:
        extracted = extract_from_sample(sample)
        if extracted is not None:
            print(extracted)
```

- We're missing particles
- We have have objets with embedded clauses, which we probably dont' want so let's ditch these as we do subjects containing a verb
- While we're at it, let's also remove anything that includes punct for now
- It's probably a good idea to avoid conjunctions in both subjects and objects since we want to use a conjunction to glue them with a non-idiom afterwards, this would only add confusion
- Also ensure subject spans are fully to left, and objects to the right of the verb spans
  - Subjects can end up included in verb spans in interrogatives “Can you hold the line?”.

And finally, let's use a named tuple to make these more usable afterwards

```python
class ProcessedSample(NamedTuple):
    verb: spacy.tokens.token.Token
    subj_span: spacy.tokens.span.Span
    verb_span: spacy.tokens.span.Span
    obj_spans: list[spacy.tokens.span.Span]
    
    def render(self) -> str:
        return " ".join(s.text for s in (self.subj_span, self.verb_span, *(span for span in self.obj_spans)))
    

def extract_from_sample(sample):
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]
    if target_verb.dep_ in {"acl", "advcl", "relcl", "acomp", "ccomp", "pcomp", "xcomp"}:
        return None
    # Again relying on that nice positional syntax
    verb_span_start = min((t.i for t in target_verb.lefts if t.dep_ == "aux"), default=target_verb.i)
    verb_span_end = max((t.i for t in target_verb.rights if t.dep_ == "prt"), default=target_verb.i)
    verb_span = sent[verb_span_start:verb_span_end+1]

    if (subj := next((t for t in target_verb.lefts if t.dep_ == "nsubj"), None)) is None:
        return None
    subj_subtree = list(subj.subtree)
    if subj_subtree[-1].i > verb_span_start:
        return None
    # NOTE: conjunctions shoud have pos CC but have CCONJ for some reason
    if any(t for t in subj_subtree if t.pos_ in {"CCONJ", "PUNCT", "VERB"}):
        return None
    subj_span = sent[subj_subtree[0].i:subj_subtree[-1].i+1]

    objects = [t for t in target_verb.rights if t.dep_ in ("dative", "dobj", "prep", "oprd")]
    has_minimal_obj = False
    obj_spans = []
    for obj in objects:
        obj_subtree = list(obj.subtree)
        if obj_subtree[0].i < verb_span_end:
            return None
        if any(t for t in obj_subtree if t.pos_ in {"CCONJ", "PUNCT", "VERB"}):
            return None
        if obj.dep_ == "dobj":
            has_minimal_obj = True
        obj_spans.append(sent[obj_subtree[0].i:obj_subtree[-1].i+1])
    if not has_minimal_obj:
        return None
    
    return ProcessedSample(verb=target_verb, subj_span=subj_span, verb_span=verb_span, obj_spans=obj_spans)
```

Final generalization

```python
processed_idiomatic_samples = defaultdict(list)
for verb, verb_idioms in rich.progress.track(idioms.items()):
    if verb not in non_idioms:
        continue
    for idiom, samples in verb_idioms.items():
        for sample in samples:
            extracted = extract_from_sample(sample)
            if extracted is not None:
                processed_idiomatic_samples[verb].append(extracted)
sum(len(l) for l in processed_idiomatic_samples.values())
```

It's probably a good idea to sample some of these to get a better idea of how usable they are.

```python
random.sample([s.render() for l in processed_idiomatic_samples.values() for s in l], k=16)
```

Wait there's more!


After checking with the Conductor, we got additional constraints:

- At most article+adj+noun for subj and obj syntagms
- Try to control the lemma frequency of the words

So let's try that

```python
def extract_from_sample(sample):
    sent = nlp(sample["context"][2])
    target_verb = sent.char_span(*sample["offsets"][0])[0]
    if target_verb.dep_ in {"acl", "advcl", "relcl", "acomp", "ccomp", "pcomp", "xcomp"}:
        return None
    verb_span_start = min((t.i for t in target_verb.lefts if t.dep_ == "aux"), default=target_verb.i)
    verb_span_end = max((t.i for t in target_verb.rights if t.dep_ == "prt"), default=target_verb.i)
    verb_span = sent[verb_span_start:verb_span_end+1]

    if (subj := next((t for t in target_verb.lefts if t.dep_ == "nsubj"), None)) is None:
        return None
    subj_subtree = list(subj.subtree)
    if subj_subtree[-1].i > verb_span_start:
        return None
    if any(t for t in subj_subtree if t.dep_ not in {"det", "amod", "nsubj"}):
        return None
    subj_span = sent[subj_subtree[0].i:subj_subtree[-1].i+1]

    objects = [t for t in target_verb.rights if t.dep_ in ("dative", "dobj", "prep", "oprd")]
    has_minimal_obj = False
    obj_spans = []
    for obj in objects:
        obj_subtree = list(obj.subtree)
        if obj_subtree[0].i < verb_span_end:
            return None
        # Problème ici: si on dégage des compléments on nique la valence verbale
        if any(t for t in obj_subtree if t.dep_ not in {"det", "amod", "dative", "dobj", "pobj", "prep", "oprd"}):
            return None
        # We want to avoid relational nouns
        if any(t for t in obj_subtree if t.dep_  == "prep" and t.head is not target_verb):
            return None
        if obj.dep_ == "dobj":
            has_minimal_obj = True
        obj_spans.append(sent[obj_subtree[0].i:obj_subtree[-1].i+1])
    if not has_minimal_obj:
        return None
    
    return ProcessedSample(verb=target_verb, subj_span=subj_span, verb_span=verb_span, obj_spans=obj_spans)
```

```python
processed_idiomatic_samples = defaultdict(list)
for verb, verb_idioms in rich.progress.track(idioms.items()):
    if verb not in non_idioms:
        continue
    for idiom, samples in verb_idioms.items():
        for sample in samples:
            extracted = extract_from_sample(sample)
            if extracted is not None:
                processed_idiomatic_samples[verb].append(extracted)
sum(len(l) for l in processed_idiomatic_samples.values())
```

```python
random.sample([s.render() for l in processed_idiomatic_samples.values() for s in l], k=16)
```

## Getting compatible non-idioms


Now the objective is, given an idiomatic example, to find non-idiomatic examples that can be glued to it. The process of selection will be largely similar.

```python
def get_corresp_nonidiom(idom_sample: ProcessedSample, non_idioms_dict: dict[str, list[dict[str, Any]]]) -> list[ProcessedSample]:
    if (compatible_non_idiom_samples := non_idioms_dict.get(idom_sample.verb.text)) is None:
        return []
    return [extract_from_sample(sample) for sample in compatible_non_idiom_samples]
```

```python
compatible = []
for idiomatic_sample in random.sample([s for l in processed_idiomatic_samples.values() for s in l], k=16):
    print(idiomatic_sample.render())
    print([s.render() for s in get_corresp_nonidiom(idiomatic_sample, non_idioms) if s is not None])
```

```python
samples_with_compatible = []
for idiomatic_sample in rich.progress.track([l for v in processed_idiomatic_samples.values() for l in v]):
    compat = [s for s in get_corresp_nonidiom(idiomatic_sample, non_idioms) if s is not None]
    if compat:
        samples_with_compatible.append((idiomatic_sample, compat))
sum(len(c) for _, c in samples_with_compatible)
```

```python
for sample, compat in random.sample(samples_with_compatible, k=16):
    c = random.choice(compat)
    print(sample.render(), "and", " ".join(s.text for s in c.obj_spans))
```

Let's try more aggressive filtering:

- Removing objects that are reduced to a single pronoun
- Ensure that the verb particles are the same in both idiom and non-idiom
  - That avoids merging e.g. “Rogers put a foot wrong” and “it couldn't put down any roots” into “Rogers put a foot wrong and any roots”
- Avoid having noun lemmas in common between objects

```python
def get_corresp_nonidiom(idiom_sample: ProcessedSample, non_idioms_dict: dict[str, list[dict[str, Any]]]) -> list[ProcessedSample]:
    if any([t.pos_ for t in obj_span] == ["PRON"] for obj_span in idiom_sample.obj_spans):
        return []
    if (compatible_non_idiom_samples := non_idioms_dict.get(idiom_sample.verb.text)) is None:
        return []
    idiom_sample_prt = [t.text for t in idiom_sample.verb_span if t.dep_ == "prt"]
    idiom_sample_noun_lemmas = {t.lemma_.lower() for s in idiom_sample.obj_spans for t in s if t.pos_ in {"NOUN", "PROPN"}}
    res = []
    for sample in compatible_non_idiom_samples:
        processed_non_idiom = extract_from_sample(sample) 
        if processed_non_idiom is None:
            continue
        if any([t.pos_ for t in obj_span] == ["PRON"] for obj_span in processed_non_idiom.obj_spans):
            return []
        sample_prt = [t.text for t in processed_non_idiom.verb_span if t.dep_ == "prt"]
        if sample_prt != idiom_sample_prt:
            continue
        sample_noun_lemmas = {t.lemma_.lower() for s in processed_non_idiom.obj_spans for t in s if t.pos_ in {"NOUN", "PROPN"}}
        if sample_noun_lemmas.intersection(idiom_sample_noun_lemmas):
            continue
        res.append(processed_non_idiom)
    return res
```

```python
samples_with_compatible = []
for idiomatic_sample in rich.progress.track([l for v in processed_idiomatic_samples.values() for l in v]):
    compat = [s for s in get_corresp_nonidiom(idiomatic_sample, non_idioms) if s is not None]
    if compat:
        samples_with_compatible.append((idiomatic_sample, compat))
sum(len(c) for _, c in samples_with_compatible)
```

```python
for sample, compat in random.sample(samples_with_compatible, k=16):
    c = random.choice(compat)
    print(sample.render(), "and", " ".join(s.text for s in c.obj_spans))
```

Something that looks jarring to me is how we deal with negations. We need different conjunctions depending on the negation status

- A negated verb span works better with *or*
- A negated second object works with *but* if neither the verb nor the first object are negated

```python
def merge(main_sample: ProcessedSample, additional_sample: ProcessedSample) -> str:
    if any(t for t in main_sample.verb_span if t.dep_ == "neg"):
        conj = "or"
    elif any(t for s in additional_sample.obj_spans for t in s if t.text in {"no", "not"}):
        if any(t for s in main_sample.obj_spans for t in s if t.text in {"no", "not"}):
            conj = "and"
        else:
            conj = "but"
    else:
        conj = "and"
    return " ".join([sample.render(), conj, *(s.text for s in c.obj_spans)])
```

```python
for sample, compat in random.sample(samples_with_compatible, k=16):
    c = random.choice(compat)
    print(merge(sample, c))
```

Here's a sandbox for diagnostics

```python
for sample, compat in samples_with_compatible:
    for c in compat:
        zeugma = merge(sample, c)
        if zeugma == "We could really make waves but no move":
            display([(t.text,t.lemma_,  t.pos_, t.dep_, t.head) for t in sample.verb.doc])
            display([(t.text,t.lemma_,  t.pos_, t.dep_, t.head) for t in c.verb.doc])
            print("\n")
```

## TL;DR lol


At this point, we have this many generated samples

```python
sum(len(c) for _, c in samples_with_compatible)
```

Here, see how they look

```python
for sample, compat in random.sample(samples_with_compatible, k=16):
    c = random.choice(compat)
    print(merge(sample, c))
```
