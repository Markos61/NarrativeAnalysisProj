import chardet
import fitz  # PyMuPDF
import docx


def extract_text(uploaded_file):
    name = uploaded_file.name.lower()

    # ---------- TXT ----------
    if name.endswith(".txt"):
        raw = uploaded_file.read()
        encoding = chardet.detect(raw)["encoding"] or "utf-8"
        return raw.decode(encoding, errors="replace")

    # ---------- PDF ----------
    elif name.endswith(".pdf"):
        text = []
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    # ---------- DOCX ----------
    elif name.endswith(".docx"):
        document = docx.Document(uploaded_file)
        return "\n".join(p.text for p in document.paragraphs)

    else:
        return "❌ Неподдерживаемый формат файла"
