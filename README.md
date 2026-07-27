# Neural FizzBuzz 🧠

**A neural network takes the world's most famous coding interview question — and scores 98/100 on numbers it has never seen.**

FizzBuzz is the most-memed interview question in software. Instead of writing the three `if` statements, this project **trains a PyTorch network to learn the rule from raw binary digits**.

## How it works

- **Training data:** numbers 101–1023, each encoded as a 10-bit binary vector
- **Test data (the interview):** numbers 1–100 — completely held out, never seen during training
- **Model:** plain MLP, `10 → 256 → 128 → 4` (classes: number / fizz / buzz / fizzbuzz)
- **Training:** full-batch Adam, cross-entropy loss, 2000 epochs, CPU-only, under a minute

## Result (real run, log included)

```
epoch 2000 | loss 0.0002 | train acc 100.0% | interview acc 98.0%

FINAL SCORE: 98/100 on numbers it has NEVER seen
misses:
  24: said '24', correct is 'fizz'
  40: said '40', correct is 'buzz'
```

The full unedited training log is in [`training_log.txt`](training_log.txt).

The two misses are the honest, interesting part: divisibility by 3 is genuinely hard to compute from binary bits (there's no single bit that tells you), so the network has to learn a real arithmetic pattern — and it *almost* perfectly does.

## Run it yourself

```bash
pip install torch
python neural_fizzbuzz.py
```

No GPU needed. Fixed seed (`42`) so you can reproduce the exact run.

## Files

| File | What it is |
|------|-----------|
| `neural_fizzbuzz.py` | The full model + training + interview, one file |
| `training_log.txt` | Unedited output of the actual training run |
