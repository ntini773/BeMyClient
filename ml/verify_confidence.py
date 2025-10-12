import numpy as np
def combined_confidence(p1, p2, w_agree=0.4, w_cert=0.6):
    """
    Combines two model churn probabilities into a single confidence score.
    p1, p2: arrays or floats (probabilities)
    Returns: confidence score (same shape as inputs)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    # 1. Agreement term
    agreement = 1 - np.abs(p1 - p2)
    print(agreement)
    # 2. Certainty term
    p_avg = (p1 + p2) / 2
    print(p_avg)
    c = 4 * (p_avg - 0.5)**2
    print(c)
    # 3. Weighted combination
    confidence = w_agree * agreement + w_cert * c

    # Clip to [0,1]
    confidence = np.clip(confidence, 0, 1)
    p_final = np.maximum(p1, p2)
    return confidence, p_final

print(combined_confidence(0.3,0.7))