"""Neural FizzBuzz — a neural network takes the world's most famous coding interview.

FizzBuzz: for each number print "fizz" if divisible by 3, "buzz" if divisible
by 5, "fizzbuzz" if divisible by both, otherwise the number itself.

Instead of writing if/else rules, this script TRAINS a small PyTorch network
to learn the rule from data:

  * Training set : numbers 101..1023, encoded as 10-bit binary vectors
  * Test set     : numbers 1..100 (the actual interview!) — never seen in training
  * Model        : 10 -> 256 -> 128 -> 4 MLP (4 classes: number/fizz/buzz/fizzbuzz)

Run it:  python neural_fizzbuzz.py
Everything runs on CPU in under a minute.
"""

import torch
import torch.nn as nn

NUM_BITS = 10          # binary encoding width (covers up to 1023)
TRAIN_START = 101      # train on 101..1023, keep 1..100 as the unseen "interview"
TRAIN_END = 2 ** NUM_BITS
EPOCHS = 2000
LR = 0.001
SEED = 42

LABELS = ["<number>", "fizz", "buzz", "fizzbuzz"]


def fizzbuzz_class(n: int) -> int:
    """Return the ground-truth class for n: 0=number, 1=fizz, 2=buzz, 3=fizzbuzz."""
    if n % 15 == 0:
        return 3
    if n % 5 == 0:
        return 2
    if n % 3 == 0:
        return 1
    return 0


def encode_binary(n: int) -> list[float]:
    """Encode integer n as a NUM_BITS-wide binary vector (LSB first)."""
    return [float((n >> i) & 1) for i in range(NUM_BITS)]


def decode(n: int, cls: int) -> str:
    """Turn a predicted class back into the FizzBuzz answer for n."""
    return str(n) if cls == 0 else LABELS[cls]


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(NUM_BITS, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )


def main() -> None:
    torch.manual_seed(SEED)

    # ----- data -----
    train_nums = list(range(TRAIN_START, TRAIN_END))
    x_train = torch.tensor([encode_binary(n) for n in train_nums])
    y_train = torch.tensor([fizzbuzz_class(n) for n in train_nums])

    test_nums = list(range(1, 101))
    x_test = torch.tensor([encode_binary(n) for n in test_nums])
    y_test = torch.tensor([fizzbuzz_class(n) for n in test_nums])

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("=" * 62)
    print("  NEURAL FIZZBUZZ — the interview begins")
    print(f"  train: {len(train_nums)} numbers ({TRAIN_START}..{TRAIN_END - 1})"
          f" | test: 1..100 (unseen)")
    print(f"  model: 10 -> 256 -> 128 -> 4 | epochs: {EPOCHS}")
    print("=" * 62)

    # ----- training loop -----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(1) == y_train).float().mean().item()
                test_acc = (model(x_test).argmax(1) == y_test).float().mean().item()
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"train acc {train_acc * 100:5.1f}% | "
                  f"interview acc {test_acc * 100:5.1f}%")

    # ----- the interview: numbers 1..100, never seen in training -----
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(1)

    print("\nTHE INTERVIEW — model answers for 1..100:")
    answers = [decode(n, int(c)) for n, c in zip(test_nums, preds)]
    for row in range(0, 100, 10):
        print("  " + " ".join(f"{a:>8}" for a in answers[row:row + 10]))

    wrong = [(n, decode(n, int(p)), decode(n, fizzbuzz_class(n)))
             for n, p in zip(test_nums, preds) if int(p) != fizzbuzz_class(n)]
    score = 100 - len(wrong)

    print(f"\nFINAL SCORE: {score}/100 on numbers it has NEVER seen")
    if wrong:
        print("misses:")
        for n, got, want in wrong:
            print(f"  {n}: said {got!r}, correct is {want!r}")
    else:
        print("Perfect interview. The network learned FizzBuzz from raw bits.")


if __name__ == "__main__":
    main()
