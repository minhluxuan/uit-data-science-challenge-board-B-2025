import os
import argparse
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from torch.nn import functional as F
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import f1_score, accuracy_score


n_models = 35

# ===== 1. LOAD DATA =====
parser = argparse.ArgumentParser(description="Create ensemble submission")
parser.add_argument("--orig", action="store_true", help="Use orig_results instead of results")
args = parser.parse_args()

root = "./orig_results" if args.orig else "./results"

label2id = {"no": 0, "intrinsic": 1, "extrinsic": 2}
id2label = {v: k for k, v in label2id.items()}

# Load test logits from all models
all_test_logits = []
subfolders = sorted(
    [sf for sf in os.listdir(root) if os.path.isdir(os.path.join(root, sf))],
    key=lambda x: int(x)
)
subfolders = [sf for sf in subfolders if int(sf) < n_models]

print(f"Loading logits from {len(subfolders)} models...")
for sf in subfolders:
    test_logits = torch.load(f"{root}/{sf}/test-logits.pt", weights_only=False)
    all_test_logits.append(test_logits)

# Convert to probabilities
test_probabilities = []
for logits in all_test_logits:
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    test_probabilities.append(probs)

print(f"Loaded {len(test_probabilities)} models, each with {test_probabilities[0].shape}")


def create_enhanced_features(probabilities_list, optimal_weights):
    n_samples = len(probabilities_list[0])
    features = []

    # 1. Base probabilities (35 × 3 = 105 features)
    base_probs = np.concatenate(probabilities_list, axis=1)

    # 2. Voting features
    votes_features = []
    for i in range(n_samples):
        preds = [np.argmax(p[i]) for p in probabilities_list]
        votes = Counter(preds)
        # One-hot encode votes
        vote_vec = [votes.get(j, 0) / len(probabilities_list) for j in range(3)]
        votes_features.append(vote_vec)
    votes_features = np.array(votes_features)

    # 3. Confidence features
    confidences = np.array([[np.max(p[i]) for p in probabilities_list]
                            for i in range(n_samples)])
    conf_stats = np.column_stack([
        confidences.mean(axis=1),
        confidences.std(axis=1),
        confidences.max(axis=1),
        confidences.min(axis=1)
    ])

    # 4. Agreement features
    agreement = np.array([
        len(set([np.argmax(p[i]) for p in probabilities_list]))
        for i in range(n_samples)
    ]).reshape(-1, 1)

    # 5. Weighted probabilities for each class
    weighted_probs = np.zeros((n_samples, 3))
    for i in range(n_samples):
        sample_probs = [p[i] for p in probabilities_list]
        weighted_probs[i] = np.average(sample_probs, axis=0, weights=optimal_weights)

    # Combine all features
    all_features = np.hstack([
        base_probs,           # 105 features
        votes_features,       # 3 features
        conf_stats,          # 4 features
        agreement,           # 1 feature
        weighted_probs       # 3 features
    ])

    return all_features
print("Creating final submission with best stacking model...")

with open(os.path.join(root, "optimal_weights.pkl"), "rb") as f:
    best_weights_de = pickle.load(f)

with open(os.path.join(root, "final_stacking_model.pkl"), "rb") as f:
    final_model = pickle.load(f)

meta_X = create_enhanced_features(test_probabilities, best_weights_de)

final_predictions = final_model.predict(meta_X)

final_labels = [id2label[pred] for pred in final_predictions]

submit_df_original = pd.read_csv(os.path.join(root, "0", "submit.csv"))
ids = submit_df_original["id"].tolist()

submission = pd.DataFrame({
    'id': ids,
    'predict_label': final_labels
})

submission.to_csv('submit.csv', index=False)