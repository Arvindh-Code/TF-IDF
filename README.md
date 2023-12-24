# TF-IDF-for-Presidential-Debates

This Python script implements an information retrieval system for analyzing and searching through a collection of presidential debate documents. The system computes and utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to measure the importance of terms within the documents, allowing for efficient document retrieval based on user queries.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- NLTK library

You can install NLTK using the following command:

```bash
pip install nltk
```

### Running the Code

1. Clone the repository:

```bash
git clone https://github.com/your_username/presidential-debate-ir.git
cd presidential-debate-ir
```

2. Ensure your presidential debate text files are stored in the `presidential_debates` directory.

3. Run the script:

```bash
python information_retrieval.py
```

## Code Overview

The code performs the following tasks:

1. **Text Processing:**
   - Reads presidential debate documents from the specified directory.
   - Tokenizes, converts to lowercase, removes stop words, and applies stemming to the text.

2. **TF-IDF Computation:**
   - Computes TF (Term Frequency) and IDF (Inverse Document Frequency) for each term in the corpus.
   - Calculates weighted TF-IDF for each document.

3. **Document Indexing:**
   - Builds a posting list for efficient document retrieval.

4. **Query Processing:**
   - Processes user queries by computing the TF-IDF for query terms.
   - Ranks documents based on the cosine similarity between the query and document vectors.

## Example Usage

```python
# Example 1: Get the TF-IDF weight for a term in a specific document
print("%.12f" % getweight("2012-10-03.txt", "health"))

# Example 2: Perform a query and retrieve relevant documents
print("(%s, %.12f)" % query("health insurance wall street"))
```

## Query Guidelines

- The system supports queries with multiple terms.
- Use meaningful terms for more accurate results.

## Notes

- Ensure that your text files are properly formatted and stored in the `presidential_debates` directory.
- The system uses TF-IDF for ranking documents, providing relevant results based on query terms.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to customize the README to include more details or specific instructions based on your requirements.
