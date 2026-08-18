# Leakage Audit

Status: raw schema inspected on 2026-08-18; modeling decision required before training.

## Highest Priority Rule

If a field cannot be proven available at checkout time, it is excluded from modeling.

## Split Policy

- Prefer the official train/test split from ASOS GraphReturns.
- Do not let test labels or future test events contribute to historical aggregate features.
- Historical aggregate features must be computed from events prior to the predicted event.

## Raw Schema Finding

The released `event_table_training.p` and `event_table_testing.p` files contain only:

- `hash(variantID)`
- `hash(customerId)`
- `isReturned`

No timestamp, checkout time, purchase sequence, or edge creation order field is present.

This means true prior-history features cannot be reconstructed event-by-event from the event table alone without validating an external ordering assumption. The customer and product node tables already include aggregate counts/rates and return-code distributions, but these may encode labels from the same split.

## Initial Exclusion Candidates

These categories are excluded unless raw inspection proves they are available before purchase:

- Return reason or top return reason tied to the current event.
- Customer-level return counts/rates if computed over the full split including the current event.
- Product-level return counts/rates if computed over the full split including the current event.
- Return-code aggregate fields if they summarize post-return outcomes.
- Any return processing timestamp.
- Post-purchase customer service outcomes.
- Delivery or fulfillment outcome fields not known at checkout.
- Any direct duplicate or encoding of the return label.

## Fields Marked Unsafe Until Proven Otherwise

| Field pattern | Source | Risk |
| --- | --- | --- |
| `returnsPerCustomer` | customer nodes | May include the target event label. |
| `customerReturnRate` | customer nodes | May include the target event label. |
| `customerId_level_return_code_*` | customer nodes | Post-return outcome aggregation risk. |
| `returnsPerProduct` | product nodes | May include the target event label. |
| `productReturnRate` | product nodes | May include the target event label. |
| `variantID_level_return_code_*` | product nodes | Post-return outcome aggregation risk. |

## Lower-Risk Candidate Fields

These fields are not automatically safe, but are plausible checkout-available candidates:

| Field pattern | Source | Notes |
| --- | --- | --- |
| `yearOfBirth`, `isMale`, `shippingCountry`, `premier` | customer nodes | Static or profile fields. |
| `productType`, `brandDesc`, `avgGbpPrice`, `avgDiscountValue` | product nodes | Product metadata/price fields. |
| `salesPerCustomer`, `salesPerProduct` | node tables | Still aggregate fields; may be allowed only if interpreted as prior purchase exposure and not label-derived. |

## Approved Modeling Tracks

User decision on 2026-08-18: use a dual-track strategy.

### `strict_no_leak`

Used for product demo and primary MVP claims.

Included fields:

- Customer: `yearOfBirth`, `isMale`, `shippingCountry`, `premier`
- Product: `productType`, `brandDesc`, `avgGbpPrice`, `avgDiscountValue`

Excluded fields:

- All return-derived customer aggregates.
- All return-derived product aggregates.
- All return-code distributions.
- Anonymous customer, variant, product, and supplier identifiers as model features.

### `paper_feature_baseline`

Used only as a comparison baseline.

It joins the official customer and product node features to event labels and records leakage-risk columns in the processed dataset manifest. It must not be presented as strictly checkout-safe unless future evidence proves the official aggregates were computed only from prior information.

## Missing Metadata Handling

User decision on 2026-08-18: use strategy C.

- Modeling keeps all event rows.
- Missing node-derived categorical values are filled with `Unknown`.
- Missing node-derived numeric values are filled with `0`.
- Each feature with missing values receives a `__missing` indicator.
- The dataset builder adds `has_complete_metadata`.
- Recommendation candidates and stable demo scenarios must be selected only from rows where `has_complete_metadata == True`.

## Feature Construction Controls

- User historical purchase count and return rate use prior events only.
- Product variant historical purchase count and return rate use prior events only.
- User-brand and user-product-type aggregates use prior events only.
- Country-product-type aggregates use prior events only.
- Low-count aggregates use smoothing or fallback to broader groups.

Given the missing timestamp, these controls require either a validated ordering assumption or a reduced modeling scope that avoids event-history features.

## Documentation Required After Inspection

For each removed field, record:

- Field name.
- Source file.
- Reason removed.
- Whether the field is direct label leakage, post-outcome leakage, future aggregate leakage, or unavailable-at-checkout risk.
