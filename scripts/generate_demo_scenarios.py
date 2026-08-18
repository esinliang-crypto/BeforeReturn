from __future__ import annotations

from src.inference.scenarios import generate_demo_scenarios, write_demo_scenarios


def main() -> None:
    scenarios = generate_demo_scenarios()
    output_path = write_demo_scenarios(scenarios)
    print(f"Wrote {output_path}")
    for scenario in scenarios:
        print(
            scenario["id"],
            f"risk={scenario['risk_probability']:.3f}",
            f"confidence={scenario['confidence']:.3f}",
            f"alternative={scenario['alternative'] is not None}",
        )


if __name__ == "__main__":
    main()

