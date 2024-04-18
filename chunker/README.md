## Semantic Chunker with chunk size thresholds

### Introduction

Langchain recently added support for Semantic Chunker, which allows you to split your documents into chunks based on the semantic similarity of its sentences. 

The two links below do a way better job at explaining how this technique works:

https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/

A limitation of LangChain's implementation though, at least at the time of this writing, is that the only criteria used by the chunker is the semantic similarity between the sentences in your text. So, technically, it's possible that some of the chunks will be larger than what the context window of your LLM can accommodate. Plus, there's also no lower bound limit to the chunk size, so you can end up with chunks with just a couple of words (document titles and headers, for example).

This custom implementation of the Semantic Chunker uses Langchain's implementation as a base, and expands it to allow the split to respect the upper and lower size bounds defined by the user. All credits for the original implementation go to Greg Kamradt and the LangChain team.

### How to use it

Initialize the Chunker by passing the embedding service you would like to use, and optionally the maximum and minimum chunk sizes:

```python
html_text_splitter = CustomSemanticChunker(EMBEDDING_SERVICE_TO_USE, breakpoint_threshold_type="percentile",
                                                   max_chunk_size=MAX_TEXT_CHUNK_SIZE, min_chunk_size=MIN_TEXT_CHUNK_SIZE)

html_text_snippets = html_text_splitter.split_documents(html_texts)
```
