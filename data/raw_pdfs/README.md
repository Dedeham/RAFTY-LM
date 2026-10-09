# Source PDFs

Place your PDF corpus in this directory. Any domain works: technical manuals, textbooks, legal documents, internal docs — whatever the pipeline should learn to answer questions about.

Set `dataset.domain_label` in `config.yaml` to describe your subject area (used in the training system prompt).

PDF files themselves are gitignored (`data/raw_pdfs/*.pdf`) — do not commit documents you don't have the rights to redistribute.

