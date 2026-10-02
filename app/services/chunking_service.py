from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


def create_chunks(pages: list[dict]):

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(
            page["content"]
        )

        for chunk in page_chunks:

            if not chunk.strip():
                continue

            chunks.append({
                "content": chunk,
                "page_number": page["page_number"]
            })

    return chunks