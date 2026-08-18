# BeforeReturn Case Study

Status: draft outline.

## Problem

Fashion ecommerce returns are costly and frustrating. BeforeReturn estimates return risk before checkout and intervenes only when model confidence, risk threshold, recommendation availability, risk reduction, and prompt frequency policy all allow it.

## Approach

The MVP uses ASOS GraphReturns historical anonymous customer-product purchase events. It treats return prediction as an event-level supervised learning problem rather than a causal intervention study.

## Product Principle

The model informs a policy. It does not directly decide what a customer should buy.

## Data Limitation

The dataset includes users with at least one return, so results are not representative of ASOS-wide behavior.

## Demo Scenarios

Pending selection after model inference artifacts are available.

