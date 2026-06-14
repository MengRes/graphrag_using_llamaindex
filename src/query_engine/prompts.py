"""Query prompts aligned with Microsoft GraphRAG local/global search."""

MAP_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
 "points": [
 {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
 {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
 ]
}}

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

Limit your response length to {max_length} words.

---Data tables---

{context_data}
"""

REDUCE_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer.

Limit your response length to {max_length} words.

---Target response length and format---

{response_type}

---Analyst Reports---

{report_data}
"""

LOCAL_SEARCH_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:
"This is an example sentence supported by data references [Data: Sources (ids), Entities (ids), Relationships (ids), Reports (ids)]."

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""

AUTO_ROUTE_PROMPT = """
Classify the user question for GraphRAG search routing.

- "global": broad, thematic, dataset-wide questions (themes, trends, comparisons across documents, summaries of the whole corpus).
- "local": entity-specific, detail questions (who is X, what does Y do, relationships of a specific entity, facts about one topic).

Reply with exactly one word: global or local.

Question: {query}
"""

NO_DATA_ANSWER = (
    "I am sorry but I am unable to answer this question given the provided data."
)

DEFAULT_RESPONSE_TYPE = "Multiple Paragraphs"
