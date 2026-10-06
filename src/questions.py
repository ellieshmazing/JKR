"""Gate questions sent to TypeSafe Jev in stage 1 of classify_jkr.py."""

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which single category best describes the post?",
        "criteria": {
            "gender_politics": "Sex, gender identity, trans rights, or women's single-sex spaces.",
            "uk_politics": "UK parties, government, legislation, courts, or elections.",
            "books_and_writing": "Books, writing, publishing, Harry Potter, or the Strike novels.",
            "personal_or_other": "Personal chat, banter, or anything else.",
        },
    },
    "hostile": {"type": "noul", "instructions": "Is the post hostile or insulting toward a person or group?"},
    "mentions_trans_people": {"type": "noul", "instructions": "Does the post mention trans people or gender identity?"},
}
