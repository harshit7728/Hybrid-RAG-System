import fitz


def extract_pdf_pages(pdf_bytes: bytes):

    pages = []

    with fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            if text.strip():

                pages.append({
                    "page_number": page_number,
                    "content": text
                })

    return pages