import re
from typing import Optional, List
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker, BreakpointThresholdType

GROUPED_SENTENCES_CHUNKING_MINIMUM_SIZE = 200


class ThresholdSemanticChunker(SemanticChunker):
    """
    Custom version of the SemanticChunker introduced by Greg Kamradt to Langchain. All credits of the original work go to him
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    The original version of this chunker doesn't take into consideration the length of the chunks, which can lead to some
    chunks being too large or small.
    This custom version adds the ability to define a minimum and maximum chunk size. The split will be based on two criteria:
        - chunk is larger than the max chunk size: in that case, even if the following sentences are still semantically similar
        to the current chunk, a split is done. This is to avoid creating chunks that are larger than the LLM's context length.
        - semantic distance between the current and next sentences is larger than the distance threshold, and the chunk size is
        larger than the minimum threshold. The minimum threshold is in place to avoid small chunks, with just a couple of words for example.
    Both max and minimum sizes are optional. If not set, then this class will work the same as the original semantic chunker.
    """

    def __init__(
            self,
            embeddings: Embeddings,
            add_start_index: bool = False,
            breakpoint_threshold_type: BreakpointThresholdType = "percentile",
            breakpoint_threshold_amount: Optional[float] = None,
            number_of_chunks: Optional[int] = None,
            max_chunk_size: int = None,
            min_chunk_size: int = None

    ):
        super().__init__(
            embeddings=embeddings,
            add_start_index=add_start_index,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks
        )
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_text(
        self,
        text: str
    ) -> List[str]:
        # Splitting the essay on '.', '?', '!', break lines
        single_sentences_list = re.split(r"(?<=[.?!])\s+|\n+", text)

        grouped_sentences = []
        group = single_sentences_list[0]
        for sentence in single_sentences_list[1:]:
            group += f" {sentence}"
            if len(group) >= GROUPED_SENTENCES_CHUNKING_MINIMUM_SIZE:
                grouped_sentences.append(group.strip())
                group = str()

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(grouped_sentences) == 1:
            return grouped_sentences
        distances, sentences = self._calculate_sentence_distances(grouped_sentences)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
        else:
            breakpoint_distance_threshold = self._calculate_breakpoint_threshold(distances)

        sentences_sizes = [len(sentence) for sentence in grouped_sentences]
        size_so_far = sentences_sizes[0]
        chunks = []
        chunk = [sentences[0]["sentence"]]

        if not self.max_chunk_size:
            self.max_chunk_size = sum(sentences_sizes)
        if not self.min_chunk_size:
            self.min_chunk_size = 1
        # Sentences are grouped based on either of two conditions: distance between them, or the length/size of the chunk.
        # If chunk is larger than the max threshold OR if the distance between the sentences is greater than the
        # distance_threshold and the size of the chunk is larger than the minimum threshold,
        # then that chunk is stored, and a new chunk is created
        for i, (sentence, sentence_size, distance) in enumerate(zip(sentences[1:], sentences_sizes[1:], distances)):
            larger_than_max = (size_so_far + sentence_size) >= self.max_chunk_size
            larger_than_min = (size_so_far + sentence_size) >= self.min_chunk_size
            distance_larger_than_threshold = distance > breakpoint_distance_threshold
            if larger_than_max or (larger_than_min and distance_larger_than_threshold):
                chunks.append(" ".join(chunk))
                size_so_far = 0
                chunk = []
            else:
                chunk.append(sentence['sentence'])
                size_so_far += sentence_size

        if chunk:
            chunks.append(" ".join(chunk))

        return chunks
