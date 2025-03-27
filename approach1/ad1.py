import numpy as np
import torch
import torch.nn as nn

# 1️⃣ Training sentences
sentences = [
    "cat chases dog",
    "dog runs fast",
    "cat likes milk",
    "dog barks loud",
    "people love dogs",
    "cats hunt mice",
]

# 2️⃣ Tokenization (mapping words to numbers)
# Build vocabulary from all sentences
words = set()
for s in sentences:
    words.update(s.split())
vocab = {word: idx for idx, word in enumerate(sorted(words))}
reverse_vocab = {idx: word for word, idx in vocab.items()}
print("Vocabulary:", vocab)

# Tokenize all sentences
tokenized_sentences = []
for s in sentences:
    tokens = [vocab[word] for word in s.split()]
    tokenized_sentences.append(tokens)
print("Tokenized sentences:", tokenized_sentences)

# 3️⃣ Creating word embeddings
embedding_dim = 5  # Increased dimension for better representation
np.random.seed(42)  # for reproducibility
embedding_matrix = np.random.randn(len(vocab), embedding_dim)
word_vectors = {word: embedding_matrix[idx] for word, idx in vocab.items()}

# 4️⃣ Word vectors
for word, vector in word_vectors.items():
    print(f"{word}: {vector}")

# 5️⃣ Convert tokens to vectors
# This will now be done during training


# 6️⃣ Neural network (simple language model)
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_dim)
        # Get the last time step output
        output = self.fc(lstm_out[:, -1, :])  # (batch_size, vocab_size)
        return output


# 7️⃣ Prepare training data
def prepare_training_data(tokenized_sentences):
    X, y = [], []
    for tokens in tokenized_sentences:
        for i in range(1, len(tokens)):
            # Use tokens up to i-1 as input, token i as target
            X.append(tokens[:i])
            y.append(tokens[i])
    return X, y


X, y = prepare_training_data(tokenized_sentences)
print("Training examples:", len(X))
print("Sample X:", X[:3])
print("Sample y:", y[:3])

# 8️⃣ Initialize network
vocab_size = len(vocab)
embedding_dim = 5
hidden_dim = 10
model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_dim)


# 9️⃣ Training loop
def train_model(model, X, y, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # Prepare input sequence
            seq = torch.tensor(X[i], dtype=torch.long).unsqueeze(0)  # Add batch dimension
            target = torch.tensor([y[i]], dtype=torch.long)

            # Forward pass
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")


# Train the model
train_model(model, X, y)


# 1️⃣0️⃣ Next word prediction function
def predict_next_word(model, sentence, vocab, reverse_vocab):
    # Tokenize the input
    tokens = [vocab.get(word, 0) for word in sentence.split()]
    seq = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # Get prediction
    with torch.no_grad():
        output = model(seq)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()

    # Get top 3 predictions with probabilities
    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
    top_words = [
        (reverse_vocab[idx.item()], prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])
    ]

    return reverse_vocab[predicted_idx], top_words


# Test prediction with different examples
test_phrases = ["cat chases", "dog runs", "cat", "dog", "people love"]

print("\nNext Word Predictions:")
for phrase in test_phrases:
    next_word, top_words = predict_next_word(model, phrase, vocab, reverse_vocab)
    print(f"Input: '{phrase}'")
    print(f"Predicted next word: '{next_word}'")
    print(f"Top 3 predictions: {top_words}")
    print("---")
