


```python
from duallaat import DualLAAT

# Load the pretrained model
model = DualLAAT.from_pretrained('tmp_ckpt')

# Define candidate ICD-10 codes
candidate_codes = {
    "I10": "Essential (primary) hypertension",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "F41.1": "Generalized anxiety disorder",
    "M54.5": "Low back pain"
}

# ============================================
# Sample Clinical Note 1
# Ground truth codes: I10, M54.5
# ============================================
note1 = """
Patient is a 62-year-old male presenting for routine follow-up. He has a 
history of essential hypertension, currently managed with lisinopril 10mg 
daily with good control. Blood pressure today is 128/82. Patient also reports 
chronic low back pain that has been present for the past 6 months, worse with 
prolonged sitting. Pain is managed with NSAIDs and physical therapy. No 
radiating symptoms or neurological deficits noted on examination.
"""

# ============================================
# Sample Clinical Note 2
# Ground truth codes: E11.9, J44.9
# ============================================
note2 = """
68-year-old female with known Type 2 diabetes mellitus presents with increased 
shortness of breath over the past week. Patient has a history of COPD and 
reports worsening dyspnea with minimal exertion. Blood glucose today is 
156 mg/dL. Lung examination reveals decreased breath sounds bilaterally with 
prolonged expiratory phase. Spirometry shows FEV1 of 58% predicted. Prescribed 
albuterol inhaler and adjusted diabetes medications.
"""

# Make predictions
predictions = model.predict(
    notes_to_code=[note1, note2],
    codes_to_consider=list(candidate_codes.values()),
)

# Extract predicted codes based on probability threshold, which defaults to 0.5
probs = predictions['probabilities']
threshold = 0.5

predicted_codes = [
    [code for code, prob in zip(candidate_codes.keys(), prob_list) if prob > threshold]
    for prob_list in probs
]

# Display results
print("=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)
print(f"\nNote 1 - Predicted Codes: {predicted_codes1}")
print(f"         Ground Truth:     ['I10', 'M54.5']")
print(f"\nNote 2 - Predicted Codes: {predicted_codes2}")
print(f"         Ground Truth:     ['E11.9', 'J44.9']")
print("=" * 60)

# Output:
# ============================================================
# PREDICTION RESULTS
# ============================================================
#
# Note 1 - Predicted Codes: ['I10', 'M54.5']
#          Ground Truth:     ['I10', 'M54.5']
#
# Note 2 - Predicted Codes: ['E11.9', 'J44.9']
#          Ground Truth:     ['E11.9', 'J44.9']
# ============================================================
```

