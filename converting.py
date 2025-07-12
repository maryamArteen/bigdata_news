import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF
from rdflib.namespace import XSD

# Load CSV
df = pd.read_csv('final_combined_results.csv')

# Create RDF graph
g = Graph()
EX = Namespace("http://example.org/misinfo#")
g.bind("ex", EX)

# Define classes
ARTICLE = EX.Article
TRUTHLABEL = EX.TruthLabel
SUBJECT = EX.Subject
TOPIC = EX.Topic
ENTITY = EX.Entity

# Define properties
hasLabel = EX.hasLabel
hasSubject = EX.hasSubject
hasTopic = EX.hasTopic
title = EX.title
topicTerms = EX.topicTerms
hasEntity = EX.hasEntity
entityName = EX.entityName
entityType = EX.entityType

# Iterate through each row
for idx, row in df.iterrows():
    article_uri = EX[f"article_{idx}"]
    g.add((article_uri, RDF.type, ARTICLE))
    g.add((article_uri, title, Literal(row['title'], datatype=XSD.string)))
    g.add((article_uri, topicTerms, Literal(row['topic_terms'], datatype=XSD.string)))

    # Add object properties
    label_uri = EX[f"label_{row['label'].strip().upper()}"]
    g.add((article_uri, hasLabel, label_uri))
    g.add((label_uri, RDF.type, TRUTHLABEL))

    subject_uri = EX[f"subject_{row['subject'].strip().lower()}"]
    g.add((article_uri, hasSubject, subject_uri))
    g.add((subject_uri, RDF.type, SUBJECT))

    if not pd.isna(row['dominant_topic']):
        topic_uri = EX[f"topic_{int(row['dominant_topic'])}"]
        g.add((article_uri, hasTopic, topic_uri))
        g.add((topic_uri, RDF.type, TOPIC))

    # Add named entities
    if isinstance(row['entities_str'], str):
        entities = [e.strip() for e in row['entities_str'].split(';') if '(' in e and ')' in e]
        for ent_idx, ent in enumerate(entities):
            try:
                name, etype = ent.rsplit('(', 1)
                name = name.strip()
                etype = etype.replace(')', '').strip()
                entity_uri = EX[f"article_{idx}_entity_{ent_idx}"]
                g.add((entity_uri, RDF.type, ENTITY))
                g.add((entity_uri, entityName, Literal(name, datatype=XSD.string)))
                g.add((entity_uri, entityType, Literal(etype, datatype=XSD.string)))
                g.add((article_uri, hasEntity, entity_uri))
            except ValueError:
                continue  # skip malformed entries

# Save to Turtle file
g.serialize("articles_data2.ttl", format="turtle", encoding="utf-8")
