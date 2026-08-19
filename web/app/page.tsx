"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Check,
  ChevronRight,
  Loader2,
  ShieldCheck,
  SlidersHorizontal
} from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import * as Switch from "@radix-ui/react-switch";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type TopFactor = {
  feature: string;
  impact: number;
  direction: string;
};

type Alternative = {
  variant_id: string;
  product_type: string;
  brand: string;
  risk_probability: number;
  relative_risk_change: number;
  reason: string;
  candidate_type: string;
  risk_basis: string;
  inventory_status: string;
  disclaimer: string;
};

type Scenario = {
  id: string;
  label: string;
  behavior: string;
  case_type: string;
  selection_rule: string;
  user_id: string;
  variant_id: string;
  country: string;
  product_type: string;
  brand: string;
  risk_probability: number;
  prediction_margin: number;
  confidence: number;
  observed_outcome: "returned" | "not_returned";
  observed_outcome_hidden_by_default: boolean;
  observed_outcome_note: string;
  alternative: Alternative | null;
};

type Prediction = Scenario & {
  risk_level: "high" | "medium" | "low";
  should_intervene: boolean;
  top_factors: TopFactor[];
  policy_reasons: string[];
  model_version: string;
};

type Policy = {
  high_risk_threshold: number;
  min_prediction_margin: number;
  max_prompts_per_1000: number;
  allow_variant_recommendations: boolean;
  allow_product_recommendations: boolean;
  min_risk_reduction: number;
};

type PolicySimulation = {
  evaluated_checkouts: number;
  estimated_prompts: number;
  eligible_checkouts: number;
  prompt_budget: number;
  artifact_rows: number;
  prompt_coverage: number;
  recall_at_policy: number;
  precision_at_policy: number;
  false_positives: number;
  user_disturbance_rate: number;
  disclaimer: string;
};

type OverviewMetrics = {
  test_positive_rate: number;
  logistic_pr_auc: number;
  catboost_pr_auc: number;
  pr_auc_absolute_gain: number;
  pr_auc_relative_gain: number;
  recall_at_10: number;
  precision_at_10: number;
  lift_at_10: number;
  brier_score: number;
  constant_baseline_brier: number;
  brier_skill_score: number;
  roc_auc: number;
  f1: number;
  precision: number;
  recall: number;
  ece: number;
  test_sample_count: number;
  model_version: string;
  evaluation_timestamp: string;
  data_processing_version: string;
};

type ExplanationFeature = {
  feature: string;
  mean_abs_shap: number;
};

type ModelExplanations = {
  feature_set: string;
  model_path: string;
  sample_rows: number;
  complete_metadata_only: boolean;
  top_features: ExplanationFeature[];
  model_version: string;
  artifact_path: string;
};

const defaultPolicy: Policy = {
  high_risk_threshold: 0.6,
  min_prediction_margin: 0.3,
  max_prompts_per_1000: 150,
  allow_variant_recommendations: true,
  allow_product_recommendations: true,
  min_risk_reduction: 0.1
};

function pct(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function decimal(value: number, digits = 3) {
  return value.toFixed(digits);
}

function count(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function featureLabel(feature: string) {
  const labels: Record<string, string> = {
    avgDiscountValue: "Avg discount",
    avgGbpPrice: "Avg price",
    brandDesc: "Brand",
    isMale: "Gender flag",
    premier: "Premier",
    productType: "Product type",
    shippingCountry: "Country",
    yearOfBirth: "Birth year"
  };
  return labels[feature] ?? feature.replaceAll("__missing", " missing");
}

function overviewMetricCards(metrics: OverviewMetrics | null) {
  if (!metrics) {
    return [
      { label: "PR-AUC", value: "Unavailable", detail: "Baseline unavailable" },
      { label: "Recall@Top 10%", value: "Unavailable", detail: "Random 10.0% · lift unavailable" },
      { label: "Brier Score", value: "Unavailable", detail: "Baseline unavailable · BSS unavailable" },
      { label: "Test Orders", value: "Unavailable", detail: "Strict offline temporal split" }
    ];
  }

  return [
    {
      label: "PR-AUC",
      value: decimal(metrics.catboost_pr_auc),
      detail: `Baseline ${pct(metrics.test_positive_rate)} · +${decimal(metrics.pr_auc_absolute_gain)}`
    },
    {
      label: "Recall@Top 10%",
      value: pct(metrics.recall_at_10),
      detail: `Random 10.0% · ${decimal(metrics.lift_at_10, 2)}× lift`
    },
    {
      label: "Brier Score",
      value: decimal(metrics.brier_score),
      detail: `Baseline ${decimal(metrics.constant_baseline_brier)} · BSS ${decimal(metrics.brier_skill_score)}`
    },
    {
      label: "Test Orders",
      value: count(metrics.test_sample_count),
      detail: "Strict offline temporal split"
    }
  ];
}

function scenarioLabel(scenario: Scenario) {
  return scenario.label || scenario.id.replaceAll("_", " ");
}

function observedOutcomeLabel(outcome: Scenario["observed_outcome"]) {
  return outcome === "returned" ? "Returned" : "Not returned";
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

export default function Home() {
  const [tab, setTab] = useState("overview");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenarioId, setSelectedScenarioId] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [showObservedOutcome, setShowObservedOutcome] = useState(false);
  const [policy, setPolicy] = useState<Policy>(defaultPolicy);
  const [simulation, setSimulation] = useState<PolicySimulation | null>(null);
  const [overviewMetrics, setOverviewMetrics] = useState<OverviewMetrics | null>(null);
  const [modelExplanations, setModelExplanations] = useState<ModelExplanations | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const explanationData = useMemo(
    () =>
      modelExplanations?.top_features.slice(0, 8).map((feature) => ({
        feature: featureLabel(feature.feature),
        impact: feature.mean_abs_shap
      })) ?? [],
    [modelExplanations]
  );

  const selectedScenario = useMemo(
    () => scenarios.find((scenario) => scenario.id === selectedScenarioId) ?? scenarios[0],
    [scenarios, selectedScenarioId]
  );

  useEffect(() => {
    fetch(`${API_BASE}/demo-scenarios`)
      .then((response) => response.json())
      .then((data: Scenario[]) => {
        setScenarios(data);
        setSelectedScenarioId(data[0]?.id ?? "");
      })
      .catch(() => setError("API is unavailable. Start FastAPI on port 8000."));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/model-metrics`)
      .then((response) => {
        if (!response.ok) throw new Error("Overview metrics are unavailable.");
        return response.json();
      })
      .then((data: OverviewMetrics) => setOverviewMetrics(data))
      .catch(() => setOverviewMetrics(null));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/model-explanations`)
      .then((response) => {
        if (!response.ok) throw new Error("Model explanations are unavailable.");
        return response.json();
      })
      .then((data: ModelExplanations) => setModelExplanations(data))
      .catch(() => setModelExplanations(null));
  }, []);

  async function evaluateRisk() {
    if (!selectedScenario) return;
    setLoading(true);
    setError("");
    setShowObservedOutcome(false);
    try {
      const result = await postJson<Prediction>("/predict-return-risk", {
        user_id: selectedScenario.user_id,
        variant_id: selectedScenario.variant_id,
        policy
      });
      setPrediction(result);
      setTab(result.alternative ? "alternative" : "checkout");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  async function simulatePolicy(nextPolicy = policy) {
    setPolicy(nextPolicy);
    try {
      const result = await postJson<PolicySimulation>("/simulate-policy", { policy: nextPolicy });
      setSimulation(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Policy simulation failed.");
    }
  }

  useEffect(() => {
    simulatePolicy(defaultPolicy);
  }, []);

  return (
    <main className="min-h-screen bg-paper">
      <div className="mx-auto flex w-full max-w-7xl gap-6 px-6 py-6">
        <aside className="w-64 shrink-0">
          <div className="mb-6">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal">BeforeReturn</p>
            <h1 className="mt-2 text-3xl font-semibold leading-tight text-ink">
              Checkout return risk, before the order is placed.
            </h1>
          </div>
          <nav className="space-y-2">
            {[
              ["overview", Activity, "Overview"],
              ["checkout", ShieldCheck, "Checkout Simulator"],
              ["alternative", ChevronRight, "Lower-risk Peer"],
              ["policy", SlidersHorizontal, "Policy Console"]
            ].map(([id, Icon, label]) => (
              <button
                key={id as string}
                onClick={() => setTab(id as string)}
                className={cn(
                  "flex h-11 w-full items-center gap-3 rounded-md px-3 text-left text-sm",
                  tab === id ? "bg-ink text-white" : "text-ink hover:bg-wash"
                )}
              >
                <Icon className="h-4 w-4" />
                {label as string}
              </button>
            ))}
          </nav>
        </aside>

        <section className="min-w-0 flex-1 space-y-5">
          {error && (
            <div className="flex items-center gap-2 rounded-md border border-rose/30 bg-white px-4 py-3 text-sm text-rose">
              <AlertCircle className="h-4 w-4" />
              {error}
            </div>
          )}

          {tab === "overview" && (
            <div className="space-y-5">
              <div className="grid grid-cols-4 gap-4">
                {overviewMetricCards(overviewMetrics).map((metric) => (
                  <Card key={metric.label}>
                    <CardContent>
                      <p className="text-sm text-muted">{metric.label}</p>
                      <p className="mt-2 text-3xl font-semibold text-ink">{metric.value}</p>
                      <p className="mt-1 text-xs text-muted">{metric.detail}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
              <div className="rounded-md border border-line bg-white px-4 py-3 text-sm leading-6 text-muted">
                GraphReturns includes customers with at least one historical return. Reported
                probabilities describe this evaluation population and must not be interpreted as the
                ASOS-wide return rate.
              </div>
              {overviewMetrics && overviewMetrics.catboost_pr_auc <= overviewMetrics.test_positive_rate && (
                <div className="rounded-md border border-line bg-white px-4 py-3 text-sm text-muted">
                  CatBoost PR-AUC is not above the positive-rate baseline.
                </div>
              )}
              {overviewMetrics && overviewMetrics.brier_skill_score <= 0 && (
                <div className="rounded-md border border-line bg-white px-4 py-3 text-sm text-muted">
                  Brier Skill Score is not above the constant-probability baseline.
                </div>
              )}
              <Card>
                <CardHeader>
                  <CardTitle>Model Signal</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-[1.1fr_0.9fr] gap-8">
                  <div className="space-y-4">
                    <p className="max-w-2xl text-lg leading-8 text-ink">
                      BeforeReturn estimates return risk for an anonymous shopper and product variant,
                      then lets policy decide whether to show a lower-risk peer option.
                    </p>
                    <p className="text-sm leading-6 text-muted">
                      Source: ASOS GraphReturns. Risk changes are model estimates, not randomized
                      causal effects. Overview metrics are read from the latest offline evaluation
                      artifact when available.
                    </p>
                    <Button onClick={() => setTab("checkout")}>
                      Try a checkout scenario
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="space-y-3">
                    <div className="h-72">
                      {modelExplanations ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={explanationData} layout="vertical" margin={{ left: 24 }}>
                            <CartesianGrid stroke="#eee8e0" horizontal={false} />
                            <XAxis type="number" hide />
                            <YAxis dataKey="feature" type="category" width={112} tickLine={false} axisLine={false} />
                            <Tooltip formatter={(value) => decimal(Number(value), 3)} />
                            <Bar dataKey="impact" fill="#0f766e" radius={[0, 4, 4, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="flex h-full items-center justify-center rounded-md border border-dashed border-line text-sm text-muted">
                          Unavailable
                        </div>
                      )}
                    </div>
                    <p className="text-xs leading-5 text-muted">
                      {modelExplanations
                        ? `SHAP summary artifact: ${modelExplanations.artifact_path} · model ${modelExplanations.model_version}`
                        : "SHAP summary artifact unavailable · model unavailable"}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {tab === "checkout" && (
            <Card>
              <CardHeader>
                <CardTitle>Checkout Simulator</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-[360px_1fr] gap-6">
                <div className="space-y-4">
                  <label className="block text-sm font-medium text-ink" htmlFor="scenario">
                    Scenario
                  </label>
                  <select
                    id="scenario"
                    value={selectedScenarioId}
                    onChange={(event) => {
                      setSelectedScenarioId(event.target.value);
                      setPrediction(null);
                      setShowObservedOutcome(false);
                    }}
                    className="h-11 w-full rounded-md border border-line bg-white px-3 text-sm"
                  >
                    {scenarios.map((scenario) => (
                      <option key={scenario.id} value={scenario.id}>
                        {scenarioLabel(scenario)}
                      </option>
                    ))}
                  </select>
                  {selectedScenario && (
                    <div className="space-y-3 rounded-md border border-line bg-wash p-4 text-sm leading-6">
                      <div>
                        <p className="font-medium text-ink">{scenarioLabel(selectedScenario)}</p>
                        <p className="text-muted">{selectedScenario.behavior}</p>
                      </div>
                      <div>
                        <p>User {selectedScenario.user_id}</p>
                        <p>Variant {selectedScenario.variant_id}</p>
                        <p>{selectedScenario.country}</p>
                        <p>{selectedScenario.brand} · {selectedScenario.product_type}</p>
                      </div>
                      <div className="flex items-center justify-between rounded-md border border-line bg-white px-3 py-2">
                        <div>
                          <p className="text-xs text-muted">Observed outcome</p>
                          <p className="font-medium text-ink">
                            {showObservedOutcome
                              ? observedOutcomeLabel(selectedScenario.observed_outcome)
                              : "Hidden"}
                          </p>
                        </div>
                        <Button
                          type="button"
                          variant="secondary"
                          onClick={() => setShowObservedOutcome((visible) => !visible)}
                        >
                          {showObservedOutcome ? "Hide outcome" : "Reveal outcome"}
                        </Button>
                      </div>
                      {showObservedOutcome && (
                        <p className="text-xs leading-5 text-muted">
                          {selectedScenario.observed_outcome_note}
                        </p>
                      )}
                    </div>
                  )}
                  <Button onClick={evaluateRisk} disabled={loading || !selectedScenario}>
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
                    Evaluate return risk
                  </Button>
                </div>
                <RiskPanel prediction={prediction} />
              </CardContent>
            </Card>
          )}

          {tab === "alternative" && <AlternativePanel prediction={prediction} onBack={() => setTab("checkout")} />}

          {tab === "policy" && (
            <PolicyConsole policy={policy} simulation={simulation} onChange={simulatePolicy} />
          )}
        </section>
      </div>
    </main>
  );
}

function RiskPanel({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) {
    return (
      <div className="flex min-h-[360px] items-center justify-center rounded-lg border border-dashed border-line text-sm text-muted">
        Risk evaluation will appear here.
      </div>
    );
  }

  const riskColor = prediction.risk_level === "high" ? "text-rose" : prediction.risk_level === "low" ? "text-teal" : "text-[#9a6a00]";
  const predictionMargin = prediction.prediction_margin ?? prediction.confidence;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Metric label="Estimated return risk" value={pct(prediction.risk_probability)} className={riskColor} />
        <Metric label="Risk level" value={prediction.risk_level} className="capitalize" />
        <Metric label="Prediction margin" value={pct(predictionMargin)} />
      </div>
      <div className="rounded-md border border-line bg-white p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium">
          {prediction.should_intervene ? <Check className="h-4 w-4 text-teal" /> : <AlertCircle className="h-4 w-4 text-muted" />}
          {prediction.should_intervene ? "Intervention allowed by policy" : "No intervention under current policy"}
        </div>
        <ul className="space-y-2 text-sm text-muted">
          {prediction.policy_reasons.map((reason) => (
            <li key={reason}>{reason.replace("pass: ", "").replace("fail: ", "")}</li>
          ))}
        </ul>
      </div>
      <div className="rounded-md border border-line bg-white p-4">
        <p className="mb-3 text-sm font-medium text-ink">Why this was flagged</p>
        <div className="space-y-2">
          {prediction.top_factors.map((factor) => (
            <div key={factor.feature} className="flex items-center justify-between text-sm">
              <span>{factor.feature}</span>
              <span className={factor.impact >= 0 ? "text-rose" : "text-teal"}>{factor.direction}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AlternativePanel({ prediction, onBack }: { prediction: Prediction | null; onBack: () => void }) {
  if (!prediction?.alternative) {
    return (
      <Card>
        <CardContent className="flex items-center justify-between">
          <p className="text-sm text-muted">No lower-risk peer option is available for the current scenario.</p>
          <Button variant="secondary" onClick={onBack}>Back to checkout</Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Lower-Risk Peer Option</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <CompareCard
            title="Original choice"
            variantId={prediction.variant_id}
            brand={prediction.brand}
            productType={prediction.product_type}
            risk={prediction.risk_probability}
          />
          <CompareCard
            title="Lower-risk peer option"
            variantId={prediction.alternative.variant_id}
            brand={prediction.alternative.brand}
            productType={prediction.alternative.product_type}
            risk={prediction.alternative.risk_probability}
          />
        </div>
        <div className="grid grid-cols-[220px_1fr] gap-4 rounded-md border border-line bg-white p-4">
          <div>
            <p className="text-xs text-muted">Estimated risk difference</p>
            <p className="mt-2 text-2xl font-semibold text-teal">
              -{pct(prediction.alternative.relative_risk_change)}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted">Peer option basis</p>
            <p className="mt-2 text-sm leading-6 text-ink">
              {prediction.alternative.reason} Candidate type: {prediction.alternative.candidate_type}.
              {` ${prediction.alternative.risk_basis}`}
            </p>
          </div>
        </div>
        <div className="rounded-md border border-line bg-wash p-4 text-sm leading-6 text-muted">
          {prediction.alternative.inventory_status} {prediction.alternative.disclaimer}
        </div>
        <div className="flex gap-3">
          <Button><Check className="h-4 w-4" /> Choose peer option</Button>
          <Button variant="secondary">Keep original choice</Button>
        </div>
      </CardContent>
    </Card>
  );
}

function PolicyConsole({
  policy,
  simulation,
  onChange
}: {
  policy: Policy;
  simulation: PolicySimulation | null;
  onChange: (policy: Policy) => void;
}) {
  function update(next: Partial<Policy>) {
    onChange({ ...policy, ...next });
  }

  function updatePeerRecommendations(value: boolean) {
    update({
      allow_variant_recommendations: value,
      allow_product_recommendations: value
    });
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Policy Console</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-[380px_1fr] gap-6">
        <div className="space-y-5">
          <Slider label="High-risk threshold" value={policy.high_risk_threshold} min={0.4} max={0.9} step={0.01} onChange={(value) => update({ high_risk_threshold: value })} />
          <Slider label="Minimum prediction margin" value={policy.min_prediction_margin} min={0} max={0.8} step={0.01} onChange={(value) => update({ min_prediction_margin: value })} />
          <Slider label="Max prompts per 1,000 checkouts" value={policy.max_prompts_per_1000} min={0} max={300} step={10} onChange={(value) => update({ max_prompts_per_1000: value })} />
          <Toggle
            label="Allow lower-risk peer options"
            checked={policy.allow_variant_recommendations || policy.allow_product_recommendations}
            onChange={updatePeerRecommendations}
          />
          <p className="text-xs leading-5 text-muted">
            Peer options are same-brand, same-product-type historical peers only.
            Candidate risk is model-estimated under the current user's checkout fields.
            Inventory is not verified.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-3 self-start">
          <Metric label="Estimated prompts" value={String(simulation?.estimated_prompts ?? 0)} />
          <Metric label="Eligible checkouts" value={String(simulation?.eligible_checkouts ?? 0)} />
          <Metric label="Prompt coverage" value={pct(simulation?.prompt_coverage ?? 0)} />
          <Metric label="Recall@Policy" value={pct(simulation?.recall_at_policy ?? 0)} />
          <Metric label="Precision" value={pct(simulation?.precision_at_policy ?? 0)} />
          <Metric label="False positives" value={String(simulation?.false_positives ?? 0)} />
          <Metric label="User disturbance rate" value={pct(simulation?.user_disturbance_rate ?? 0)} />
          <Metric label="Artifact rows" value={count(simulation?.artifact_rows ?? 0)} />
          <p className="col-span-2 mt-2 text-sm leading-6 text-muted">{simulation?.disclaimer}</p>
        </div>
      </CardContent>
    </Card>
  );
}

function Metric({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="rounded-md border border-line bg-white p-4">
      <p className="text-xs text-muted">{label}</p>
      <p className={cn("mt-2 text-2xl font-semibold text-ink", className)}>{value}</p>
    </div>
  );
}

function CompareCard({
  title,
  variantId,
  brand,
  productType,
  risk
}: {
  title: string;
  variantId: string;
  brand: string;
  productType: string;
  risk: number;
}) {
  return (
    <div className="rounded-md border border-line bg-white p-4">
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="mt-3 break-all text-xs text-muted">Variant {variantId}</p>
      <p className="mt-3 text-sm text-muted">{brand} · {productType}</p>
      <p className="mt-4 text-3xl font-semibold text-ink">{pct(risk)}</p>
      <p className="mt-1 text-xs text-muted">Estimated return risk</p>
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 flex items-center justify-between text-sm text-ink">
        {label}
        <span className="text-muted">{max <= 1 ? pct(value) : value}</span>
      </span>
      <input
        className="w-full accent-teal"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-ink">{label}</span>
      <Switch.Root
        checked={checked}
        onCheckedChange={onChange}
        className="relative h-6 w-11 rounded-full bg-line data-[state=checked]:bg-teal"
      >
        <Switch.Thumb className="block h-5 w-5 translate-x-0.5 rounded-full bg-white transition-transform data-[state=checked]:translate-x-5" />
      </Switch.Root>
    </div>
  );
}
