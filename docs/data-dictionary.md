# Data Dictionary

Status: raw schema inspected on 2026-08-18; semantic availability still requires leakage decisions.

## Source Files

The official OSF project exposes pickled files for training and testing splits:

| File | Expected role | Commit policy |
| --- | --- | --- |
| `event_table_training.p` | Training purchase/return edges | Do not commit |
| `event_table_testing.p` | Testing purchase/return edges | Do not commit |
| `customer_nodes_training.p` | Training customer node features | Do not commit |
| `customer_nodes_testing.p` | Testing customer node features | Do not commit |
| `product_nodes_training.p` | Training product variant node features | Do not commit |
| `product_nodes_testing.p` | Testing product variant node features | Do not commit |

Observed file sizes total approximately 742 MB.

## Observed Shapes

| File | Rows | Columns |
| --- | ---: | ---: |
| `event_table_training.p` | 1,369,133 | 3 |
| `event_table_testing.p` | 1,460,366 | 3 |
| `customer_nodes_training.p` | 777,001 | 30 |
| `customer_nodes_testing.p` | 825,598 | 30 |
| `product_nodes_training.p` | 411,495 | 44 |
| `product_nodes_testing.p` | 411,544 | 44 |

## Event Tables

| Column | Type | Notes |
| --- | --- | --- |
| `hash(variantID)` | int64 | Product variant key. |
| `hash(customerId)` | int64 | Anonymous customer key. |
| `isReturned` | int64 | Binary target label; positive class is returned. |

No event timestamp or purchase sequence column is present in the event table.

## Customer Node Tables

| Column group | Examples | Notes |
| --- | --- | --- |
| Identity key | `hash(customerId)` | Anonymous customer key. |
| Demographics/profile | `yearOfBirth`, `isMale`, `shippingCountry`, `premier` | Candidate checkout-available features, subject to semantics. |
| Customer aggregates | `salesPerCustomer`, `returnsPerCustomer`, `customerReturnRate` | Leakage risk if computed using current or future labels. |
| Return-code aggregates | `customerId_level_return_code_*` | High leakage risk unless proven to use only prior returns. |
| Country one-hot fields | `Country_A` ... `Country_I` | Redundant with `shippingCountry`; low leakage risk. |

The raw customer files contain a duplicate column name: `customerId_level_return_code_D`.

## Product Node Tables

| Column group | Examples | Notes |
| --- | --- | --- |
| Identity keys | `hash(variantID)`, `hash(productID)`, `hash(supplierRef)` | Anonymous product, variant, and supplier keys. |
| Product metadata | `productType`, `brandDesc`, `avgGbpPrice`, `avgDiscountValue` | Candidate features if available by checkout time. |
| Product aggregates | `salesPerProduct`, `returnsPerProduct`, `productReturnRate` | Leakage risk if computed using current or future labels. |
| Return-code aggregates | `variantID_level_return_code_*` | High leakage risk unless proven to use only prior returns. |
| Brand/product type one-hot fields | `Brand_*`, `productType_*` | Redundant with categorical fields; low leakage risk if static. |

The raw product files contain a duplicate column name: `variantID_level_return_code_D`.

## Target Label

Target: `isReturned`, where `1` is returned and `0` is not returned.

## Prediction Time Boundary

The model must only use information available at checkout time. Fields that can only be known after purchase, fulfillment, return processing, or customer service outcomes are excluded.

The raw event tables do not expose timestamps. This prevents true event-by-event prior-history reconstruction from the released files alone unless another hidden ordering convention is validated.

## Pending Inspection

- Join keys between event, customer, and product variant tables.
- Whether the provided split is purely temporal.
- Fields requiring removal for leakage risk.
- Categorical cardinality and low-frequency smoothing needs.
