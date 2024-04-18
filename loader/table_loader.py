import json
from io import StringIO
from typing import Any, List
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from pandas import MultiIndex
import logging

logger = logging.getLogger(__name__)

TABLE_TO_JSON_START_ROW_TAG = "<START_ROW>"
TABLE_TO_JSON_END_ROW_TAG = "</END_ROW>"
TABLE_TO_JSON_SPLITTER_DELIMITER = "<DELIMITER>"


class TableWebLoader(WebBaseLoader):

    known_caption_classes: List = ['pTableCaptionCMT']

    @staticmethod
    def __build_initial_metadata(soup: Any, url: str) -> dict:
        """Build preliminary metadata from BeautifulSoup output."""
        metadata = {"source": url}
        title = soup.find("title")
        metadata["title"] = title.get_text() if title else "No title found"
        description = soup.find("meta", attrs={"name": "description"})
        metadata["description"] = description.get("content", "No description found.") if description else "No description found."
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")
        return metadata

    @staticmethod
    def __generate_text_from_table_row(row):
        """
        Given a row from a table, generate a text to better represent the information in it, in an LLM-friendly format
        """

        row_text = f"{TABLE_TO_JSON_START_ROW_TAG}\n"

        # MultiIndex series can't be converted to dict and then dumped to string, as the dict keys are tuples
        # Need to first convert the keys from tuples to string
        if isinstance(row.keys(), MultiIndex):
            row_dict = {str(k): v for k, v in row.to_dict().items()}
        else:
            row_dict = row.to_dict()

        row_text += json.dumps(row_dict, indent=2)
        row_text += f'\n{TABLE_TO_JSON_END_ROW_TAG}\n{TABLE_TO_JSON_SPLITTER_DELIMITER}'
        return row_text

    def __extract_caption_from_table(self, table: BeautifulSoup) -> str:
        """Search for the caption of an HTML table"""

        caption = table.find('caption')

        if caption:
            return caption.get_text()

        for caption_class in self.known_caption_classes:
            # If table doesn't have a caption, tries to find the previous caption in the document.
            # Some documents have the table caption outside of the table object, and its own class
            caption = table.find_previous('p', {'class': caption_class})
            if not caption:
                continue
            # If previous caption is close to the table, they are likely related and the caption is returned
            if (table.sourceline - caption.sourceline) <= 10:
                return caption.get_text()

        # If all attempts failed, then return empty string
        return str()

    def aload(self, index_table_as_raw: bool = False) -> (List[Document], List[Document]):
        """
        Load the web documents and extract both table and text. Tables are extracted as JSON and optionally as raw text;
        everything else in the web document is extracted as raw text (similar to langchain's original WebBaseLoader class).
        :param index_table_as_raw: boolean that defines if the table contents will also be parsed as raw text, in addition
        to the JSON parsing.
        """

        results = self.scrape_all(self.web_paths)

        text_docs = []
        table_docs = []
        for path, soup in zip(self.web_paths, results):

            tables = soup.findAll('table')

            # For all tables in the HTML file, read them as a pandas dataframe. Then convert each row to text, preserving
            # the headers and cell values
            for table in tables:
                metadata = self.__build_initial_metadata(soup, path)
                caption = self.__extract_caption_from_table(table)
                if not table:
                    continue
                if table.tbody:
                    rows = table.tbody.find_all('tr')
                else:
                    rows = table.find_all('tr')
                if rows and len(rows) >= 2:
                    try:
                        table_dfs = pd.read_html(StringIO(str(table)), flavor='bs4')
                        if not table_dfs:
                            continue
                        table_df = table_dfs[0]
                        table_df['row_text'] = table_df.apply(lambda row: self.__generate_text_from_table_row(row), axis=1)
                        table_contents = '\n'.join(table_df['row_text'].tolist())
                        metadata['table_caption'] = caption
                        table_docs.append(Document(page_content=table_contents, metadata=metadata))
                        if not index_table_as_raw:
                            table.decompose()
                    except Exception as e:
                        logger.error(f"Error while parsing table, skipping it. Error details: {e}")

            # Extract raw, non-table text
            metadata = self.__build_initial_metadata(soup, path)
            text = soup.get_text(**self.bs_get_text_kwargs)
            text_docs.append(Document(page_content=text, metadata=metadata))

        return text_docs, table_docs
