# %%
import itertools
import spacy

# %%
text = "New York City (NYC), often called simply New York, is the most populous city in the United States. With an estimated 2019 population of 8,336,817 distributed over about 302.6 square miles (784 km2), New York City is also the most densely populated major city in the United States.[11] Located at the southern tip of the State of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban landmass.[12] With almost 20 million people in its metropolitan statistical area and approximately 23 million in its combined statistical area, it is one of the world's most populous megacities. New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce,[13] entertainment, research, technology, education, politics, tourism, art, fashion, and sports. Home to the headquarters of the United Nations,[14] New York is an important center for international diplomacy"

# %%
# spaCy for production

# Doc: https://spacy.io/api/language
nlp = spacy.load("en_core_web_sm")
# Doc: https://spacy.io/api/doc
doc = nlp(text)

# %%
# Doc: https://spacy.io/api/token
for token in itertools.islice(doc, 10):
    print(token, token.pos_, token.dep_)

# Doc: https://spacy.io/api/span
for entity in itertools.islice(doc.ents, 10):
    print(entity, entity.label_)

# %%
spacy.displacy.render(doc, style='ent')

# %%
