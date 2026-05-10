#!/usr/bin/env python3
"""
Analyze condition distribution and create coarser groupings.
"""

import pandas as pd
from collections import Counter

# Load metadata
df = pd.read_csv('/home/sankar/data/scin/real_labeled_metadata.csv')

# Get condition counts
condition_counts = Counter(df['condition'])

print("="*80)
print("CONDITION DISTRIBUTION")
print("="*80)
print(f"\nTotal images: {len(df)}")
print(f"Total unique conditions: {len(condition_counts)}")
print(f"\nTop 30 conditions:")
for condition, count in condition_counts.most_common(30):
    print(f"  {condition}: {count} images")

print(f"\nConditions with <10 images: {sum(1 for c, count in condition_counts.items() if count < 10)}")
print(f"Conditions with <50 images: {sum(1 for c, count in condition_counts.items() if count < 50)}")
print(f"Conditions with <100 images: {sum(1 for c, count in condition_counts.items() if count < 100)}")

# Create coarse category mapping based on medical groups
COARSE_CATEGORIES = {
    # Inflammatory/Eczematous (Group 1)
    'Eczema': 'Inflammatory Dermatitis',
    'Infected eczema': 'Inflammatory Dermatitis',
    'Seborrheic Dermatitis': 'Inflammatory Dermatitis',
    'Stasis Dermatitis': 'Inflammatory Dermatitis',
    'Allergic Contact Dermatitis': 'Inflammatory Dermatitis',
    'Irritant Contact Dermatitis': 'Inflammatory Dermatitis',
    'Acute and chronic dermatitis': 'Inflammatory Dermatitis',
    'Photodermatitis': 'Inflammatory Dermatitis',
    'CD - Contact dermatitis': 'Inflammatory Dermatitis',
    'Acute dermatitis, NOS': 'Inflammatory Dermatitis',
    'Chronic dermatitis, NOS': 'Inflammatory Dermatitis',
    'Intertrigo': 'Inflammatory Dermatitis',
    'Xerosis': 'Inflammatory Dermatitis',

    # Psoriasis and related (Group 2)
    'Psoriasis': 'Psoriatic Conditions',
    'Pityriasis rosea': 'Psoriatic Conditions',
    'Pityriasis rubra pilaris': 'Psoriatic Conditions',

    # Infections - Fungal (Group 3)
    'Tinea': 'Fungal Infections',
    'Tinea Versicolor': 'Fungal Infections',
    'Candida infection': 'Fungal Infections',

    # Infections - Bacterial (Group 4)
    'Impetigo': 'Bacterial Infections',
    'Folliculitis': 'Bacterial Infections',
    'Cellulitis': 'Bacterial Infections',
    'Skin infection': 'Bacterial Infections',
    'Abscess': 'Bacterial Infections',
    'Ecthyma': 'Bacterial Infections',

    # Infections - Viral (Group 5)
    'Viral Exanthem': 'Viral Infections',
    'Herpes Zoster': 'Viral Infections',
    'Herpes Simplex': 'Viral Infections',
    'Molluscum Contagiosum': 'Viral Infections',
    'Warts': 'Viral Infections',
    'Verruca vulgaris': 'Viral Infections',

    # Urticaria and allergic (Group 6)
    'Urticaria': 'Urticaria/Allergic',
    'Drug Rash': 'Urticaria/Allergic',
    'Angioedema': 'Urticaria/Allergic',
    'Hypersensitivity': 'Urticaria/Allergic',
    'Erythema multiforme': 'Urticaria/Allergic',

    # Acne and related (Group 7)
    'Acne': 'Acne/Follicular',
    'Rosacea': 'Acne/Follicular',
    'Perioral Dermatitis': 'Acne/Follicular',
    'Keratosis pilaris': 'Acne/Follicular',
    'Miliaria': 'Acne/Follicular',

    # Pigmentary disorders (Group 8)
    'Hyperpigmentation': 'Pigmentary Disorders',
    'Post-Inflammatory hyperpigmentation': 'Pigmentary Disorders',
    'Vitiligo': 'Pigmentary Disorders',
    'Melasma': 'Pigmentary Disorders',

    # Benign tumors/growths (Group 9)
    'Seborrheic Keratosis': 'Benign Tumors',
    'Lipoma': 'Benign Tumors',
    'Neurofibroma': 'Benign Tumors',
    'Dermatofibroma': 'Benign Tumors',
    'Nevus': 'Benign Tumors',
    'Cherry Angioma': 'Benign Tumors',
    'Granuloma annulare': 'Benign Tumors',

    # Malignant/Premalignant (Group 10)
    'Basal Cell Carcinoma': 'Skin Cancer',
    'SCC/SCCIS': 'Skin Cancer',
    'Melanoma': 'Skin Cancer',
    'Actinic Keratosis': 'Skin Cancer',
    "Kaposi's sarcoma of skin": 'Skin Cancer',

    # Autoimmune/Connective tissue (Group 11)
    'Lichen planus/lichenoid eruption': 'Autoimmune/Lichenoid',
    'Lichen Simplex Chronicus': 'Autoimmune/Lichenoid',
    'Lupus Erythematosus': 'Autoimmune/Lichenoid',
    'Dermatomyositis': 'Autoimmune/Lichenoid',
    'Scleroderma': 'Autoimmune/Lichenoid',
    'Cutaneous lupus': 'Autoimmune/Lichenoid',
    'Lichen nitidus': 'Autoimmune/Lichenoid',

    # Vascular (Group 12)
    'Purpura': 'Vascular Disorders',
    'Leukocytoclastic Vasculitis': 'Vascular Disorders',
    'Petechiae': 'Vascular Disorders',
    'Pigmented purpuric eruption': 'Vascular Disorders',
    'O/E - ecchymoses present': 'Vascular Disorders',

    # Parasitic/Insect (Group 13)
    'Scabies': 'Parasitic/Insect',
    'Insect Bite': 'Parasitic/Insect',
    'Dermatosis due to flea': 'Parasitic/Insect',
    'Lyme Disease': 'Parasitic/Insect',

    # Bullous (Group 14)
    'Bullous dermatitis': 'Bullous Disorders',
    'Pemphigus': 'Bullous Disorders',
    'Bullous Pemphigoid': 'Bullous Disorders',

    # Pruritic (Group 15)
    'Prurigo nodularis': 'Pruritic Conditions',

    # Trauma/Wounds (Group 16)
    'Inflicted skin lesions': 'Trauma/Wounds',
    'Scar': 'Trauma/Wounds',
    'Keloid': 'Trauma/Wounds',
    'Scar Condition': 'Trauma/Wounds',
    'Abrasion, scrape, or scab': 'Trauma/Wounds',
    'Superficial wound of body region': 'Trauma/Wounds',
}

# Apply coarse categories
df['coarse_category'] = df['condition'].map(COARSE_CATEGORIES).fillna('Other')

# Get coarse category distribution
coarse_counts = Counter(df['coarse_category'])

print("\n" + "="*80)
print("COARSE CATEGORY DISTRIBUTION")
print("="*80)
print(f"\nTotal coarse categories: {len(coarse_counts)}")
for category, count in coarse_counts.most_common():
    pct = count / len(df) * 100
    print(f"  {category}: {count} images ({pct:.1f}%)")

# Show which fine-grained conditions didn't get mapped
unmapped = df[df['coarse_category'] == 'Other']['condition'].value_counts()
if len(unmapped) > 0:
    print(f"\n⚠ Unmapped conditions ({len(unmapped)} unique):")
    for cond, count in unmapped.head(20).items():
        print(f"  {cond}: {count} images")

# Calculate average images per category
avg_per_coarse = len(df) / len([c for c in coarse_counts if c != 'Other'])
print(f"\nAverage images per coarse category (excluding Other): {avg_per_coarse:.1f}")

# Save the coarse category mapping
print("\n" + "="*80)
print("Saving coarse category metadata...")
output_path = '/home/sankar/data/scin/coarse_labeled_metadata.csv'
df.to_csv(output_path, index=False)
print(f"✓ Saved to: {output_path}")
