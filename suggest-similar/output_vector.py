import sqlite3
import numpy as np

def fetch_from_db(word):
    conn = sqlite3.connect('./suggest-similar/db/words_embeddings.db')
    cursor = conn.cursor()

    cursor.execute("SELECT vector, error FROM word_embeddings WHERE word=?", (word,))
    row = cursor.fetchone()
    
    if row:
        vector_blob, error = row
        vector = np.frombuffer(vector_blob, dtype=np.float32) if vector_blob else None
        conn.close()
        return vector, error
    else:
        conn.close()
        print(f"No entry found for the word: {word}")
        return None, None

# Example usage:
word = "東京"
vector, error = fetch_from_db(word)
print("Vector:", vector)
print("Error:", error)