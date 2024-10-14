from collections import defaultdict
import pandas as pd
import spacy
from spacy.tokens import Span


@spacy.registry.misc("entity_scrubber")
def articles_scrubber():
    def scrubber_func(span: Span) -> str:
        if span[0].ent_type_ or span[0].text in ("a", "the", "their", "every", "other", "two", "I", "it", "she", "you"):
            # ignore named entities
            return "INELIGIBLE_PHRASE"
        while len(span) > 1 and span[0].text in ("a", "the", "their", "every", "other", "two", "I", "it", "she", "you"):
            span = span[1:]
        return span.lemma_
        # return span.text
    return scrubber_func


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank", config={"scrubber": {"@misc": "entity_scrubber"}})

print(nlp.pipe_names)


df = pd.read_csv('documents/labeled_suicidewatch_posts_pushshiftapi.csv')
df = df.dropna()
chunk = 1000

phrases = defaultdict(int)
for start in range(0, len(df), chunk):
    text = ' '.join([str(row['Title'] + " " + row['Content']).lower() for _, row in df.iloc[start:start+chunk].iterrows()])
    doc = nlp(text)

    for i, phrase in enumerate(doc._.phrases):
        if phrase.text != "INELIGIBLE_PHRASE":
            phrases[str(phrase.text)] += phrase.count
            # print(i, phrase.count, phrase.text)

phrases = [(k, v) for k, v in phrases.items()]
with open('suicidal_phrases.txt', 'w', encoding='utf-8') as output:
    for phrase, count in sorted(phrases, key=lambda el: el[1], reverse=True):
        output.write(f"{count} {phrase}\n")

