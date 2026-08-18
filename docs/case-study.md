# BeforeReturn Case Study

Status: draft outline.

## Problem

Fashion ecommerce returns are costly and frustrating. BeforeReturn estimates return risk before checkout and intervenes only when prediction margin, risk threshold, recommendation availability, risk reduction, and prompt frequency policy all allow it.

## Approach

The MVP uses ASOS GraphReturns historical anonymous customer-product purchase events. It treats return prediction as an event-level supervised learning problem rather than a causal intervention study.

## Product Principle

The model informs a policy. It does not directly decide what a customer should buy.

## Data Limitation

The dataset includes users with at least one return, so results are not representative of ASOS-wide behavior.

## Demo Scenarios

Stable scenarios are generated in `data/samples/demo_scenarios.json`:

- High risk, high prediction margin, same-brand/product-type lower-risk alternative exists.
- High risk, low prediction margin, no intervention.
- Low risk, no intervention.

Alternatives are scored by combining the same anonymous user's checkout-available profile fields with candidate product metadata. The first version falls back to same brand and product type because the released data does not expose real size/color adjacency.
