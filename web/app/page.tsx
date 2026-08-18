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
};

type Scenario = {
  id: string;
  user_id: string;
  variant_id: string;
  country: string;
  product_type: string;
  brand: string;
  risk_probability: number;
  confidence: number;
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
  min_confidence: number;
  max_prompts_per_1000: number;
  allow_variant_recommendations: boolean;
  allow_product_recommendations: boolean;
  min_risk_reduction: number;
};

type PolicySimulation = {
  evaluated_checkouts: number;
  estimated_prompts: number;
  prompt_coverage: number;
  recall_at_policy: number;
  precision_at_policy: number;
  false_positives: number;
  user_disturbance_rate: number;
  disclaimer: string;
};

const defaultPolicy: Policy = {
  high_risk_threshold: 0.6,
  min_confidence: 0.3,
  max_prompts_per_1000: 150,
  allow_variant_recommendations: true,
  allow_product_recommendations: true,
  min_risk_reduction: 0.1
};

const metricCards = [
  { label: "PR-AUC", value: "0.680", detail: "strict calibrated CatBoost" },
  { label: "Recall@Top 10%", value: "0.141", detail: "offline test split" },
  { label: "Brier Score", value: "0.229", detail: "calibrated artifact" },
  { label: "Selection Bias", value: "Known", detail: "users with at least one return" }
];

const shapData = [
  { feature: "Product type", impact: 0.324 },
  { feature: "Country", impact: 0.302 },
  { feature: "Price", impact: 0.254 },
  { feature: "Brand", impact: 0.089 },
  { feature: "Birth year", impact: 0.076 }
];

function pct(value: number) {
  return `${Math.round(value * 1000) / 10}%`;
}

function scenarioLabel(id: string) {
  return id
    .replaceAll("_", " ")
    .replace("high risk high confidence with alternative", "High risk, high confidence")
    .replace("high risk low confidence no intervention", "High risk, low confidence")
    .replace("low risk no intervention", "Low risk");
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
  const [policy, setPolicy] = useState<Policy>(defaultPolicy);
  const [simulation, setSimulation] = useState<PolicySimulation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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

  async function evaluateRisk() {
    if (!selectedScenario) return;
    setLoading(true);
    setError("");
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
              ["alternative", ChevronRight, "Safer Alternative"],
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
                {metricCards.map((metric) => (
                  <Card key={metric.label}>
                    <CardContent>
                      <p className="text-sm text-muted">{metric.label}</p>
                      <p className="mt-2 text-3xl font-semibold text-ink">{metric.value}</p>
                      <p className="mt-1 text-xs text-muted">{metric.detail}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
              <Card>
                <CardHeader>
                  <CardTitle>Model Signal</CardTitle>
                </CardHeader>
                <CardContent className="grid grid-cols-[1.1fr_0.9fr] gap-8">
                  <div className="space-y-4">
                    <p className="max-w-2xl text-lg leading-8 text-ink">
                      BeforeReturn estimates return risk for an anonymous shopper and product variant,
                      then lets policy decide whether to show a lower-risk alternative.
                    </p>
                    <p className="text-sm leading-6 text-muted">
                      Source: ASOS GraphReturns. The dataset contains users with at least one return, so
                      results are not representative of ASOS-wide return behavior. Risk changes are model
                      estimates, not randomized causal effects.
                    </p>
                    <Button onClick={() => setTab("checkout")}>
                      Try a checkout scenario
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ left: 24 }}>
                        <CartesianGrid stroke="#eee8e0" horizontal={false} />
                        <XAxis type="number" hide />
                        <YAxis dataKey="feature" type="category" width={92} tickLine={false} axisLine={false} />
                        <Tooltip />
                        <Bar dataKey="impact" fill="#0f766e" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
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
                    }}
                    className="h-11 w-full rounded-md border border-line bg-white px-3 text-sm"
                  >
                    {scenarios.map((scenario) => (
                      <option key={scenario.id} value={scenario.id}>
                        {scenarioLabel(scenario.id)}
                      </option>
                    ))}
                  </select>
                  {selectedScenario && (
                    <div className="rounded-md border border-line bg-wash p-4 text-sm leading-6">
                      <p>User {selectedScenario.user_id}</p>
                      <p>Variant {selectedScenario.variant_id}</p>
                      <p>{selectedScenario.country}</p>
                      <p>{selectedScenario.brand} · {selectedScenario.product_type}</p>
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

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <Metric label="Estimated return risk" value={pct(prediction.risk_probability)} className={riskColor} />
        <Metric label="Risk level" value={prediction.risk_level} className="capitalize" />
        <Metric label="Model confidence" value={pct(prediction.confidence)} />
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
          <p className="text-sm text-muted">No lower-risk alternative is available for the current scenario.</p>
          <Button variant="secondary" onClick={onBack}>Back to checkout</Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>A Lower-Risk Alternative Is Available</CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-4">
          <CompareCard title="Original choice" brand={prediction.brand} productType={prediction.product_type} risk={prediction.risk_probability} />
          <CompareCard title="Recommended alternative" brand={prediction.alternative.brand} productType={prediction.alternative.product_type} risk={prediction.alternative.risk_probability} />
        </div>
        <div className="rounded-md border border-line bg-wash p-4 text-sm leading-6 text-muted">
          Risk change is a historical-data model estimate, not a randomized causal effect.
        </div>
        <div className="flex gap-3">
          <Button><Check className="h-4 w-4" /> Accept suggestion</Button>
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>Policy Console</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-[380px_1fr] gap-6">
        <div className="space-y-5">
          <Slider label="High-risk threshold" value={policy.high_risk_threshold} min={0.4} max={0.9} step={0.01} onChange={(value) => update({ high_risk_threshold: value })} />
          <Slider label="Minimum confidence" value={policy.min_confidence} min={0} max={0.8} step={0.01} onChange={(value) => update({ min_confidence: value })} />
          <Slider label="Max prompts per 1,000 checkouts" value={policy.max_prompts_per_1000} min={0} max={300} step={10} onChange={(value) => update({ max_prompts_per_1000: value })} />
          <Toggle label="Allow variant recommendations" checked={policy.allow_variant_recommendations} onChange={(value) => update({ allow_variant_recommendations: value })} />
          <Toggle label="Allow product recommendations" checked={policy.allow_product_recommendations} onChange={(value) => update({ allow_product_recommendations: value })} />
        </div>
        <div className="grid grid-cols-2 gap-3 self-start">
          <Metric label="Estimated prompts" value={String(simulation?.estimated_prompts ?? 0)} />
          <Metric label="Prompt coverage" value={pct(simulation?.prompt_coverage ?? 0)} />
          <Metric label="Recall@Policy" value={pct(simulation?.recall_at_policy ?? 0)} />
          <Metric label="Precision" value={pct(simulation?.precision_at_policy ?? 0)} />
          <Metric label="False positives" value={String(simulation?.false_positives ?? 0)} />
          <Metric label="User disturbance rate" value={pct(simulation?.user_disturbance_rate ?? 0)} />
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

function CompareCard({ title, brand, productType, risk }: { title: string; brand: string; productType: string; risk: number }) {
  return (
    <div className="rounded-md border border-line bg-white p-4">
      <p className="text-sm font-medium text-ink">{title}</p>
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
