# Proofread — Text Correction & Spell Checking

Context-aware spell checker and light grammar fixer. Flask API + web UI + analysis notebook.

## Run
```bash
pip install -r requirements.txt
python prepare_data.py      # downloads Brown corpus -> data/corpus.txt
python app.py               # http://localhost:5000
pytest                      # tests
```

## How it works
1. **Tokenise** keeping offsets, so every change maps back to the original text.
2. **Spelling**: unknown words get edit-distance 1–2 candidates, ranked by
   `log P(word) + 0.8 * bigram context(prev, word, next) - 3 * edit distance`.
3. **Real-word errors**: confusable sets (their/there/they're, your/you're, its/it's, ...) are swapped
   when bigram context favours the alternative by a wide margin.
4. **Grammar rules**: a/an, repeated words, standalone "i", sentence capitalisation, spacing before punctuation.

## API
- `POST /api/correct` `{"text": "..."}` -> `{original, corrected, changes[{start,end,original,corrected,type,suggestions}]}`
- `GET /api/suggest?word=recieve`
- `GET /api/eval` -> top-1 accuracy on the built-in misspelling set

## Data
Brown corpus (~1.1M words) supplies word and bigram frequencies. Swap any large plain-text file into `data/corpus.txt` to improve coverage.

## Limits
Word-level n-gram model: no long-range grammar (subject-verb agreement, tense). Proper nouns mid-sentence are skipped.
