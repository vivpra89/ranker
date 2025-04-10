                                     DCN v2 Re-Ranker Architecture
                                     ===========================

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    Input Features                                            │
├───────────┬───────────┬────────────┬──────────┬───────────┬────────────┬──────────────────┤
│  Product  │   User    │    Geo     │ Country  │   Time    │  Session   │   Interaction    │
│Embeddings │Embeddings │  Features  │ Features │  Features │  Features  │    History       │
└─────┬─────┴─────┬─────┴─────┬──────┴────┬─────┴─────┬─────┴──────┬─────┴────────┬─────────┘
      │           │           │            │           │            │              │
      ▼           ▼           ▼            ▼           ▼            ▼              ▼
┌─────────────┐ ┌───────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐ ┌─────────┐
│  Feature    │ │ Feature   │ │  Text    │ │  Text   │ │ Feature  │ │  Feature   │ │Sequence │
│ Processing  │ │Processing │ │Embedder  │ │Embedder │ │Processing│ │ Processing │ │Encoder  │
└──────┬──────┘ └─────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘ └──────┬─────┘ └────┬────┘
       │              │            │           │           │            │            │
       └──────────────┴────────────┴───────┬───┴───────────┴────────────┴────────────┘
                                          │
                                          ▼
                                ┌──────────────────────┐
                                │   Feature Fusion     │
                                │   & Normalization    │
                                └──────────┬───────────┘
                                          │
                                          ▼
                              ┌────────────────────────────┐
                              │    Combined Features       │
                              └──────────────┬─────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────┐
                              │  Geo-Specific Experts      │
                              │                            │
                              │  ┌─────┐ ┌─────┐ ┌─────┐  │
                              │  │ NA  │ │ EU  │ │ APAC│  │
                              │  └──┬──┘ └──┬──┘ └──┬──┘  │
                              │     │       │       │      │
                              │     └───────┼───────┘      │
                              │             ▼              │
                              │     Geo-Based Gating       │
                              └──────────────┬─────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────┐
                              │     Global Expert          │
                              │   (Shared Knowledge)       │
                              └──────────────┬─────────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                   ┌──────────────────┐         ┌───────────────────┐
                   │  Cross Network   │         │   Deep Network    │
                   │                  │         │                   │
                   │ Feature Crossing │         │ Non-linear       │
                   │ (Low-rank DCN)   │         │ Patterns (MLP)   │
                   └────────┬─────────┘         └────────┬────────┘
                           │                             │
                           └──────────────┬─────────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │  Feature Fusion   │
                               │  & Combination    │
                               └────────┬──────────┘
                                       │
                                       ▼
                               ┌───────────────────┐
                               │   Task Heads      │
                               └────────┬──────────┘
                                       │
                           ┌───────────┼───────────┬────────────┐
                           ▼           ▼           ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
                    │  Click   │ │ Purchase │ │ Add to   │ │ Revenue  │
                    │  Score   │ │  Score   │ │  Cart    │ │  Score   │
                    └──────────┘ └──────────┘ └──────────┘ └──────────┘


Training Strategy
===============

1. Data Partitioning:
   - Split training data by geo regions (NA, EU, APAC)
   - Maintain a global dataset combining all regions

2. Training Process:
   a) Pre-training phase:
      - Train global expert on complete dataset
      - Initialize geo-specific experts with global expert weights
   
   b) Specialized training phase:
      - Fine-tune each geo expert on region-specific data
      - Keep global expert frozen as shared knowledge base
      - Train geo-based gating network to learn optimal mixing

3. Gating Mechanism:
   - Input: Geo features + Context
   - Output: Mixing weights for experts
   - Soft gating: Allow partial contribution from multiple experts
   - Hard attention to primary geo when confidence is high

4. Loss Function:
   L_total = α * L_primary_geo + β * L_global + γ * L_other_geos
   where:
   - α: Primary geo weight (higher)
   - β: Global knowledge weight
   - γ: Cross-geo learning weight (lower)

5. Regularization:
   - L2 distance between geo experts and global expert
   - Entropy regularization on gating outputs
   - Sparse gating to encourage specialization
``` 