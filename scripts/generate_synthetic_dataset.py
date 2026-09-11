"""
Generates a synthetic multi-turn "customer-support agent with memory"
conversation dataset in the exact format run_baseline.py / run_gap_experiment.py
expect:

    [
      {"query": str, "fact_to_store": str | null, "gold_answer": str | null},
      ...
    ]

Each simulated "customer" has a few facts stated early (account details,
preferences), then later turns ask the agent to recall those facts --
this is exactly the long-horizon recall pattern LongMemEval also tests,
so this can be your drop-in placeholder while you set up (or in place of)
LongMemEval.

Usage:
    python scripts/generate_synthetic_dataset.py --n-customers 10 --out data/custom_synthetic_conversations.json
"""
import argparse
import json
import random
from pathlib import Path

FIRST_NAMES = ["Aditi", "Rohan", "Meera", "Karan", "Sneha", "Vikram", "Priya", "Arjun"]
PRODUCTS = ["Pro Plan", "Basic Plan", "Team Plan", "Enterprise Plan"]
ISSUES = ["billing error", "login issue", "feature request", "refund request", "password reset"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai"]

TOOL_CALLS = [
    "look_up_account", "check_billing_history", "search_knowledge_base", "escalate_to_human",
]


def build_customer_conversation(customer_id: int, n_turns: int = 14) -> list[dict]:
    name = random.choice(FIRST_NAMES)
    plan = random.choice(PRODUCTS)
    city = random.choice(CITIES)
    issue = random.choice(ISSUES)
    account_id = f"ACC-{1000 + customer_id}"

    turns: list[dict] = []

    # --- early turns: facts get stated (these should be written to agent memory) ---
    turns.append({
        "query": f"Hi, my name is {name} and I'm on the {plan}.",
        "fact_to_store": f"Customer name is {name}, subscribed to {plan}.",
        "gold_answer": None,
    })
    turns.append({
        "query": f"I'm based in {city}, just so you know for billing.",
        "fact_to_store": f"Customer is based in {city}.",
        "gold_answer": None,
    })
    turns.append({
        "query": f"My account ID is {account_id}.",
        "fact_to_store": f"Customer account ID is {account_id}.",
        "gold_answer": None,
    })
    turns.append({
        "query": f"I'm writing in about a {issue}.",
        "fact_to_store": f"Customer's open issue: {issue}.",
        "gold_answer": None,
    })

    # --- middle turns: simulated tool-use / retrieval turns (filler, no new facts) ---
    filler_queries = [
        "Can you check what plans are available for upgrade?",
        "How long does a refund usually take?",
        "Is there a mobile app for this?",
        "What are your support hours?",
        "Do you support two-factor authentication?",
        "Can I get an invoice emailed to me?",
    ]
    n_filler = max(0, n_turns - 4 - 3)  # leave room for 3 recall turns at the end
    for i in range(n_filler):
        turns.append({
            "query": random.choice(filler_queries),
            "fact_to_store": None,
            "gold_answer": None,
            "tool_call": random.choice(TOOL_CALLS),  # optional, useful for Gap 3 experiments
        })

    # --- final turns: recall questions, this is what you score answer_quality against ---
    turns.append({
        "query": "Just to confirm, what plan am I on?",
        "fact_to_store": None,
        "gold_answer": plan,
    })
    turns.append({
        "query": "What's my account ID again?",
        "fact_to_store": None,
        "gold_answer": account_id,
    })
    turns.append({
        "query": "What issue did I originally contact you about?",
        "fact_to_store": None,
        "gold_answer": issue,
    })

    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-customers", type=int, default=10,
                         help="number of simulated customer conversations to generate")
    parser.add_argument("--turns-per-customer", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/custom_synthetic_conversations.json")
    args = parser.parse_args()

    random.seed(args.seed)

    all_conversations = []
    for cid in range(args.n_customers):
        convo = build_customer_conversation(cid, n_turns=args.turns_per_customer)
        all_conversations.append({"conversation_id": cid, "turns": convo})

    # Flatten to a single turn-list if you want one long "session" per run,
    # OR keep per-customer for session-scoped evaluation (recommended -- matches
    # the LongMemEval-style multi-session setup described in RESEARCH_PLAN.md).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_conversations, indent=2))
    print(f"Wrote {len(all_conversations)} conversations "
          f"({sum(len(c['turns']) for c in all_conversations)} total turns) -> {out_path}")
    print("Example single-conversation turn list (pass this directly to run_baseline.py --data):")
    print(json.dumps(all_conversations[0]["turns"][:3], indent=2))


if __name__ == "__main__":
    main()
