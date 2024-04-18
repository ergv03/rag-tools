## Extract tables from HTML documents as JSON

### Introduction

Extracting tables from HTML documents can be tricky, and usually most web loaders will simply extract them as raw text, which can lead to the structure of the table being lost.

This class is an extension of LangChain's Base WebLoader, that attempts to fix this extraction problem. Please note that this class can't cover all possible scenarios, due to the all the different ways web documents can be structured. But it's a good start.

Please note that all non-table content from the web document will be parsed using Langchain's default WebLoader, and not converted to JSON.


### How to use it

```python
from loader.table_loader import TableWebLoader
 
urls = ['https://www.cisco.com/c/en/us/products/collateral/security/firepower-9000-series/datasheet-c78-742471.html']
loader = TableWebLoader(urls, continue_on_failure=True)
html_docs, table_docs = loader.aload()

print(table_docs[0].page_content)
```
And the results should look like this (the `````<DELIMITER>````` tag can be used by your splitter to split large tables into smaller batches of rows, for example):

```json
<START_ROW>
{
  "Model": "SM-40",
  "Firewall": "80G",
  "NGFW": "55G",
  "Next-Generation Intrusion Prevention System (NGIPS)": "60G",
  "Interfaces": "8 x SFP+ on-chassis",
  "Optional interfaces": "2 x NMs: 1/10/40/100G, FTW"
}
</END_ROW>
<DELIMITER>
<START_ROW>
{
  "Model": "SM-48",
  "Firewall": "80G",
  "NGFW": "65G",
  "Next-Generation Intrusion Prevention System (NGIPS)": "68G",
  "Interfaces": "8 x SFP+ on-chassis",
  "Optional interfaces": "2 x NMs: 1/10/40/100G, FTW"
}
</END_ROW>
...
```
