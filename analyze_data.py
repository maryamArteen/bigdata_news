import pandas as pd


true_df = pd.read_csv('True.csv')  
fake_df = pd.read_csv('Fake.csv')  

print("=== DATASET OVERVIEW ===")
print(f"True news articles: {len(true_df)}")
print(f"Fake news articles: {len(fake_df)}")

print("\n=== DATE RANGES ===")
print(f"True dates: {true_df['date'].min()} to {true_df['date'].max()}")
print(f"Fake dates: {fake_df['date'].min()} to {fake_df['date'].max()}")

print("\n=== SUBJECT CATEGORIES ===")
print("True subjects:")
print(true_df['subject'].value_counts())
print("\nFake subjects:")
print(fake_df['subject'].value_counts())

print("\n=== RANDOM SAMPLES FOR TESTING ===")
print("Random true titles:")
for title in true_df.sample(5)['title']:
    print(f"  - {title}")

print("\nRandom fake titles:")
for title in fake_df.sample(5)['title']:
    print(f"  - {title}")