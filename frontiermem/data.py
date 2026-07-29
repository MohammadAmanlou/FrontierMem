from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import random
import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    family: str
    domain: str
    preference_label: str
    surface_anchor: str
    stable_context: Sequence[str]
    scoped_context: Sequence[str]
    scoped_type: str
    applies_when: str
    not_when: str
    shared_queries: Sequence[str]
    crossing_queries: Sequence[str]
    uncertain_queries: Sequence[str]
    match_terms: Sequence[str]
    mismatch_terms: Sequence[str]
    clarification_question: str
    apply_response: str
    neutral_response: str


FILLERS = [
    "Earlier, we discussed a calendar reminder and a grocery list.",
    "The user also asked for help rephrasing a short technical note.",
    "A separate session involved planning a routine appointment.",
    "Another conversation concerned a book recommendation unrelated to this preference.",
    "The assistant previously helped summarize a news article.",
    "There was also a short exchange about organizing files on a laptop.",
]


SPECS: List[ScenarioSpec] = [
    ScenarioSpec(
        family="travel_budget",
        domain="travel",
        preference_label="budget-conscious accommodation",
        surface_anchor="For this trip, please find a cheap but clean hotel.",
        stable_context=[
            "I generally choose the least expensive clean hotel, even when someone else pays.",
            "Across my trips, I care more about price than luxury.",
            "Saving money on accommodation is a persistent priority for me.",
        ],
        scoped_context=[
            "I usually enjoy comfortable premium hotels.",
            "This trip is different because the university reimbursement cap is strict.",
            "The low price request is only for this capped trip, not a general travel preference.",
        ],
        scoped_type="temporary_constraint",
        applies_when="the traveler is paying personally or a strict reimbursement cap applies",
        not_when="the trip is fully reimbursed without a cap",
        shared_queries=[
            "I am paying for this trip myself and my budget is tight. Which hotel should I choose?",
            "For a self-funded weekend with a strict budget, recommend a hotel.",
            "I need accommodation that stays within my limited personal budget.",
        ],
        crossing_queries=[
            "My employer will cover the full hotel cost with no reimbursement cap. What should I book?",
            "This conference hotel is fully paid by the company, so cost is not constrained.",
            "The university confirmed unlimited hotel reimbursement for this trip.",
        ],
        uncertain_queries=[
            "Find a hotel for my next conference trip; I do not know the reimbursement details yet.",
            "Recommend accommodation for a work trip, but the payment policy is still unclear.",
            "I need a hotel for a conference and have not checked who will pay.",
        ],
        match_terms=["paying", "self-funded", "strict budget", "limited personal budget", "cap applies"],
        mismatch_terms=["no reimbursement cap", "fully paid", "unlimited hotel reimbursement", "cost is not constrained"],
        clarification_question="Will you pay personally, or is this trip reimbursed without a cap?",
        apply_response="prioritize clean, well-reviewed budget hotels",
        neutral_response="compare comfort, location, and quality without forcing the cheapest option",
    ),
    ScenarioSpec(
        family="writing_formality",
        domain="writing",
        preference_label="formal tone",
        surface_anchor="Please make this message formal.",
        stable_context=[
            "I generally prefer formal wording in written communication, even with people I know well.",
            "A polished and professional tone is my default across audiences.",
            "I consistently choose formality over casual phrasing.",
        ],
        scoped_context=[
            "I am casual and playful with friends.",
            "I use formal wording only for professors, job applications, and official requests.",
            "The formal style in this message is role-specific rather than my global preference.",
        ],
        scoped_type="contextual_preference",
        applies_when="the audience is a professor, employer, or official institution",
        not_when="the audience is a close friend or informal peer group",
        shared_queries=[
            "Draft an email to my professor requesting a meeting.",
            "Write a cover message for a job application.",
            "Help me contact the department office about a formal request.",
        ],
        crossing_queries=[
            "Write a quick message to my closest friend about dinner tonight.",
            "Help me reply in our casual friend group chat.",
            "Draft a playful note to a friend I have known for years.",
        ],
        uncertain_queries=[
            "Write a message to a colleague; I have not said how close we are.",
            "Help me contact someone from my network, but the relationship is unclear.",
            "Draft a note to an acquaintance without knowing whether the setting is formal.",
        ],
        match_terms=["professor", "job application", "department office", "formal request", "employer"],
        mismatch_terms=["closest friend", "friend group", "playful note", "casual"],
        clarification_question="Who is the recipient, and should the message sound official or casual?",
        apply_response="use polished, professional, and formally structured language",
        neutral_response="match the recipient naturally rather than imposing a formal style",
    ),
    ScenarioSpec(
        family="education_detail",
        domain="education",
        preference_label="detailed step-by-step explanations",
        surface_anchor="Explain it step by step and include the reasoning.",
        stable_context=[
            "I learn best through detailed explanations and usually want every reasoning step.",
            "Even under time pressure, I prefer understanding the full derivation.",
            "Depth is a stable learning preference for me across technical subjects.",
        ],
        scoped_context=[
            "When I am learning a new concept, I want detailed explanations.",
            "During timed quizzes, I only want the final answer so I can move quickly.",
            "The desired level of detail depends on whether I am learning or being tested.",
        ],
        scoped_type="contextual_preference",
        applies_when="the user is learning a new or unfamiliar concept",
        not_when="the user is in a timed quiz or explicitly requests only the final answer",
        shared_queries=[
            "Teach me this theorem from the beginning; it is new to me.",
            "I am learning this algorithm for the first time. Can you explain it?",
            "Help me understand a new concept rather than just giving the result.",
        ],
        crossing_queries=[
            "I am in a timed quiz. Give only the final option.",
            "This is an exam question and I need the direct answer immediately.",
            "I have thirty seconds left; just state the result.",
        ],
        uncertain_queries=[
            "Help me review this topic.",
            "Can you go over this problem with me?",
            "Explain this material, but I have not said how much detail I need.",
        ],
        match_terms=["new to me", "first time", "understand a new concept", "teach me"],
        mismatch_terms=["timed quiz", "exam question", "only the final", "thirty seconds", "just state"],
        clarification_question="Do you want a quick final answer or a full step-by-step explanation?",
        apply_response="give a structured, step-by-step explanation with explicit reasoning",
        neutral_response="provide the direct result without adding unnecessary derivation",
    ),
    ScenarioSpec(
        family="food_health",
        domain="food",
        preference_label="healthy meals",
        surface_anchor="Please suggest a healthy meal for me.",
        stable_context=[
            "I consistently prioritize nutritious meals, including on celebrations and weekends.",
            "Health is more important to me than indulgence in nearly every food decision.",
            "Choosing healthy food is a long-term preference, not a short diet phase.",
        ],
        scoped_context=[
            "I normally enjoy indulgent food on social occasions.",
            "I am following a strict healthy meal plan only until my race this Sunday.",
            "The current healthy choices are a temporary training constraint.",
        ],
        scoped_type="temporary_constraint",
        applies_when="the meal occurs during the current pre-race training period",
        not_when="the race has ended and the user is attending a celebration",
        shared_queries=[
            "It is three days before my race. What should I eat tonight?",
            "Suggest dinner during my current training week.",
            "I need a meal while I am still following the pre-race plan.",
        ],
        crossing_queries=[
            "The race is over and tonight is my birthday celebration. What should I order?",
            "Recommend food for a celebration next month, long after the training plan ends.",
            "I finished the race and want a relaxed social dinner.",
        ],
        uncertain_queries=[
            "Suggest dinner for an upcoming weekend, but I have not said whether it is before the race.",
            "What should I eat on a date whose timing relative to the race is unclear?",
            "Recommend a meal for later; the training schedule is not specified.",
        ],
        match_terms=["before my race", "training week", "pre-race plan", "three days before"],
        mismatch_terms=["race is over", "next month", "finished the race", "birthday celebration"],
        clarification_question="Is this meal during the pre-race plan or after the race has ended?",
        apply_response="recommend a nutritious meal aligned with the health goal",
        neutral_response="consider the occasion and current goal instead of assuming a strict healthy plan",
    ),
    ScenarioSpec(
        family="travel_proximity",
        domain="travel",
        preference_label="accommodation close to the destination",
        surface_anchor="Please find a hotel very close to the venue.",
        stable_context=[
            "I strongly dislike commuting and almost always choose the closest practical hotel.",
            "Location is my main travel criterion across business and leisure trips.",
            "I am willing to pay more to avoid a long daily commute.",
        ],
        scoped_context=[
            "For leisure travel, I often stay farther away to explore different neighborhoods.",
            "I need a hotel near the venue only on one-day business trips with a packed schedule.",
            "The proximity request is tied to short business travel, not all trips.",
        ],
        scoped_type="contextual_preference",
        applies_when="the trip is a short business visit with a tightly packed schedule",
        not_when="the trip is a relaxed multi-day leisure vacation",
        shared_queries=[
            "I have a one-day business trip with meetings all day. Where should I stay?",
            "Recommend a hotel for a short work visit with no commuting time available.",
            "I need accommodation for a packed overnight business schedule.",
        ],
        crossing_queries=[
            "I am taking a relaxed week-long vacation and want to explore several neighborhoods.",
            "Recommend a hotel for a long leisure trip where commuting is acceptable.",
            "This is a slow-paced holiday rather than a business visit.",
        ],
        uncertain_queries=[
            "Find a hotel for my next trip; I have not said whether it is business or leisure.",
            "Recommend accommodation without knowing the trip length or schedule.",
            "I need a place to stay, but the purpose of the trip is unclear.",
        ],
        match_terms=["one-day business", "short work visit", "packed overnight", "meetings all day"],
        mismatch_terms=["week-long vacation", "long leisure", "slow-paced holiday", "commuting is acceptable"],
        clarification_question="Is this a short, tightly scheduled business trip or a relaxed leisure trip?",
        apply_response="prioritize hotels within a very short walk of the main destination",
        neutral_response="balance neighborhood quality, price, and experience rather than forcing proximity",
    ),
    ScenarioSpec(
        family="writing_concise",
        domain="writing",
        preference_label="concise responses",
        surface_anchor="Keep the answer very short.",
        stable_context=[
            "I generally prefer concise answers across routine, academic, and professional tasks.",
            "Extra detail usually distracts me, even when there is no deadline.",
            "Brevity is a persistent interaction preference for me.",
        ],
        scoped_context=[
            "I normally appreciate explanation and context.",
            "I asked for a short answer only because an urgent deadline was minutes away.",
            "The brevity request is a temporary time constraint rather than a stable preference.",
        ],
        scoped_type="temporary_constraint",
        applies_when="the user is under an immediate deadline or explicitly requests brevity",
        not_when="there is enough time and the user is learning or planning",
        shared_queries=[
            "My meeting starts in five minutes. Summarize the result.",
            "I have an immediate deadline; give me the shortest useful answer.",
            "This is urgent, so keep the response brief.",
        ],
        crossing_queries=[
            "Help me deeply understand this research proposal; there is no deadline.",
            "I am planning for next month and want a complete explanation.",
            "Teach me this carefully when time is not a constraint.",
        ],
        uncertain_queries=[
            "Help me answer this question.",
            "Can you respond to this request?",
            "Explain this, but I have not specified the desired level of detail.",
        ],
        match_terms=["five minutes", "immediate deadline", "urgent", "keep the response brief"],
        mismatch_terms=["deeply understand", "no deadline", "complete explanation", "time is not a constraint"],
        clarification_question="Do you want a brief answer or a more complete explanation?",
        apply_response="produce a compact answer with only the essential information",
        neutral_response="use the amount of detail appropriate for the task instead of forcing brevity",
    ),
    ScenarioSpec(
        family="education_examples",
        domain="education",
        preference_label="example-driven explanations",
        surface_anchor="Please include concrete examples.",
        stable_context=[
            "I consistently understand concepts better through examples in every subject.",
            "Even in theoretical mathematics, examples help me before formal definitions.",
            "Example-first teaching is a stable preference for me.",
        ],
        scoped_context=[
            "Examples are especially useful to me for programming and applied tasks.",
            "For pure mathematics, I prefer a formal proof without illustrative examples.",
            "My preference depends on the subject rather than applying globally.",
        ],
        scoped_type="contextual_preference",
        applies_when="the topic is programming or an applied technical task",
        not_when="the task is a pure mathematics proof",
        shared_queries=[
            "Teach me this Python concept and show a working example.",
            "Explain this programming pattern in an applied coding task.",
            "Help me learn an API through a concrete code example.",
        ],
        crossing_queries=[
            "Prove this abstract algebra theorem formally.",
            "I need a rigorous pure mathematics proof.",
            "Write the formal proof without relying on an illustrative case.",
        ],
        uncertain_queries=[
            "Explain this statistics concept; I have not said whether I want examples or a formal derivation.",
            "Teach me a technical idea whose subject type is unclear.",
            "Help me understand this concept, but the desired style is unspecified.",
        ],
        match_terms=["Python", "programming", "coding", "API", "working example"],
        mismatch_terms=["abstract algebra", "pure mathematics proof", "formal proof", "without relying"],
        clarification_question="Would you prefer a concrete example or a formal derivation for this topic?",
        apply_response="lead with a concrete example and then connect it to the concept",
        neutral_response="use a formal explanation without assuming examples are desired",
    ),
    ScenarioSpec(
        family="food_vegetarian_ownership",
        domain="food",
        preference_label="vegetarian food",
        surface_anchor="Please make sure there are vegetarian options.",
        stable_context=[
            "I do not eat meat and have followed a vegetarian diet for several years.",
            "The dietary preference belongs to me personally.",
            "I need vegetarian choices whenever food is selected for me.",
        ],
        scoped_context=[
            "I eat meat and do not follow a vegetarian diet.",
            "My sister is vegetarian, so I request vegetarian options only when she joins.",
            "The preference belongs to another person rather than to me.",
        ],
        scoped_type="other_owner",
        applies_when="the meal includes the vegetarian sister",
        not_when="the meal is only for the user",
        shared_queries=[
            "Plan dinner for me and my vegetarian sister.",
            "Recommend a restaurant for a meal where my sister will join.",
            "Choose food for a family dinner that includes my vegetarian sibling.",
        ],
        crossing_queries=[
            "Suggest lunch only for me; my sister is not coming.",
            "What should I eat alone today?",
            "Recommend a meal for the user only, with no vegetarian guest.",
        ],
        uncertain_queries=[
            "Plan a group dinner, but I have not listed the guests.",
            "Recommend a restaurant for an event whose attendees are unknown.",
            "Choose food for a gathering without knowing whether my sister will join.",
        ],
        match_terms=["vegetarian sister", "sister will join", "vegetarian sibling", "includes my"],
        mismatch_terms=["only for me", "eat alone", "sister is not coming", "no vegetarian guest"],
        clarification_question="Will your vegetarian sister be part of this meal, or is it only for you?",
        apply_response="ensure the recommendation includes strong vegetarian choices",
        neutral_response="do not assume the user personally requires vegetarian food",
    ),
    ScenarioSpec(
        family="writing_directness",
        domain="writing",
        preference_label="direct wording",
        surface_anchor="Make the message direct and straightforward.",
        stable_context=[
            "I generally prefer direct communication, even in delicate situations.",
            "I value clarity over softening language across audiences.",
            "Straightforward wording is my stable communication style.",
        ],
        scoped_context=[
            "I prefer direct wording for task instructions and operational messages.",
            "For apologies or emotionally sensitive conversations, I want tactful language.",
            "Directness applies to task coordination, not every social context.",
        ],
        scoped_type="contextual_preference",
        applies_when="the message gives instructions or coordinates a concrete task",
        not_when="the message is an apology or emotionally sensitive conversation",
        shared_queries=[
            "Write instructions to a teammate about the next task.",
            "Draft an operational message assigning responsibilities.",
            "Tell a collaborator exactly what action is needed.",
        ],
        crossing_queries=[
            "Help me apologize to a close friend after an argument.",
            "Write a sensitive message to someone who is upset.",
            "Draft a compassionate response about an emotional issue.",
        ],
        uncertain_queries=[
            "Write to a colleague about a concern, but the emotional sensitivity is unclear.",
            "Help me send a message whose purpose may be corrective or supportive.",
            "Draft a note without knowing whether it is a task request or a delicate conversation.",
        ],
        match_terms=["instructions", "assigning responsibilities", "action is needed", "operational"],
        mismatch_terms=["apologize", "sensitive message", "emotional issue", "compassionate"],
        clarification_question="Is this a task-oriented instruction or a sensitive interpersonal message?",
        apply_response="state the required action clearly and directly",
        neutral_response="adapt the tone to the interpersonal situation rather than forcing bluntness",
    ),
    ScenarioSpec(
        family="travel_flexibility",
        domain="travel",
        preference_label="flexible cancellation",
        surface_anchor="Only show options with flexible cancellation.",
        stable_context=[
            "I consistently value flexible cancellation because I dislike being locked into plans.",
            "I choose refundable bookings even when my schedule looks fixed.",
            "Flexibility is a stable travel preference for me.",
        ],
        scoped_context=[
            "I normally choose cheaper non-refundable options when dates are fixed.",
            "I need flexible cancellation only when the schedule is uncertain.",
            "The request depends on plan uncertainty rather than being a global preference.",
        ],
        scoped_type="contextual_preference",
        applies_when="the dates or attendance are uncertain",
        not_when="the trip dates are fixed and guaranteed",
        shared_queries=[
            "My dates may change and I might cancel. What should I book?",
            "Recommend travel options while the schedule is still uncertain.",
            "The event is not confirmed, so I need a booking strategy.",
        ],
        crossing_queries=[
            "The wedding date is fixed and I am definitely attending.",
            "My travel dates are guaranteed and cannot change.",
            "This is a confirmed trip with a completely fixed schedule.",
        ],
        uncertain_queries=[
            "Book travel for next month; I have not said whether the dates may change.",
            "Recommend a ticket without knowing how certain the schedule is.",
            "I need a booking, but plan stability is unspecified.",
        ],
        match_terms=["may change", "schedule is still uncertain", "not confirmed", "might cancel"],
        mismatch_terms=["date is fixed", "dates are guaranteed", "completely fixed", "definitely attending"],
        clarification_question="Are the dates fixed, or is there a meaningful chance the plan will change?",
        apply_response="prioritize refundable options with flexible cancellation terms",
        neutral_response="compare price and convenience without automatically paying for flexibility",
    ),
    ScenarioSpec(
        family="food_quickprep",
        domain="food",
        preference_label="quick meal preparation",
        surface_anchor="Suggest something very quick to prepare.",
        stable_context=[
            "I generally dislike cooking and almost always want meals that take very little time.",
            "Even on weekends, I prefer minimal preparation.",
            "Fast preparation is a stable food preference for me.",
        ],
        scoped_context=[
            "On workdays I need quick meals because my schedule is crowded.",
            "On relaxed weekends, I enjoy slow and elaborate cooking projects.",
            "The quick-preparation request is limited to busy weekdays.",
        ],
        scoped_type="contextual_preference",
        applies_when="the meal is on a busy workday",
        not_when="the user has a relaxed weekend available for cooking",
        shared_queries=[
            "It is a busy Tuesday and I have twenty minutes for dinner.",
            "Suggest food for a crowded workday evening.",
            "I need dinner after a long weekday with almost no cooking time.",
        ],
        crossing_queries=[
            "It is a relaxed Sunday and I want to spend the afternoon cooking.",
            "Recommend a weekend cooking project when I have plenty of time.",
            "I am free today and want an elaborate homemade meal.",
        ],
        uncertain_queries=[
            "Suggest dinner, but I have not said whether today is busy or relaxed.",
            "What should I cook on an unspecified day?",
            "Recommend a meal without knowing how much time I have.",
        ],
        match_terms=["busy Tuesday", "workday", "twenty minutes", "almost no cooking time", "crowded"],
        mismatch_terms=["relaxed Sunday", "weekend cooking project", "plenty of time", "elaborate homemade"],
        clarification_question="Is this a busy day requiring a quick meal, or do you have time to cook?",
        apply_response="recommend a meal that can be prepared quickly with minimal steps",
        neutral_response="consider more elaborate options if the user has time and interest",
    ),
    ScenarioSpec(
        family="education_encouragement",
        domain="education",
        preference_label="encouraging feedback",
        surface_anchor="Please be encouraging in your response.",
        stable_context=[
            "I generally respond well to motivational and encouraging feedback.",
            "Even during formal evaluation, supportive phrasing helps me improve.",
            "An encouraging tone is a stable interaction preference for me.",
        ],
        scoped_context=[
            "When I am struggling with a difficult topic, encouragement helps me persist.",
            "When I ask for grading or formal evaluation, I prefer neutral and objective feedback.",
            "The need for encouragement depends on whether I am learning or being assessed.",
        ],
        scoped_type="contextual_preference",
        applies_when="the user is struggling while learning a difficult topic",
        not_when="the user requests objective grading or formal evaluation",
        shared_queries=[
            "I am stuck and discouraged by this hard topic. Help me continue.",
            "I keep failing to understand this concept and need guidance.",
            "Teach me this difficult material; I am losing confidence.",
        ],
        crossing_queries=[
            "Grade this solution objectively and list the errors.",
            "Evaluate my answer using a strict rubric.",
            "Give a neutral assessment of this submission.",
        ],
        uncertain_queries=[
            "Review my progress, but I have not said whether I want coaching or grading.",
            "Comment on my work without knowing whether this is practice or evaluation.",
            "Give feedback, but the desired role is unclear.",
        ],
        match_terms=["stuck", "discouraged", "losing confidence", "hard topic"],
        mismatch_terms=["grade", "strict rubric", "neutral assessment", "objectively"],
        clarification_question="Would you like supportive coaching or a neutral formal evaluation?",
        apply_response="use supportive language while still giving actionable guidance",
        neutral_response="provide objective feedback without adding unnecessary motivational framing",
    ),
]


def _history(spec: ScenarioSpec, profile_variant: str, rng: random.Random) -> str:
    fillers = rng.sample(FILLERS, k=2)
    if profile_variant == "stable":
        core = list(spec.stable_context)
    else:
        core = list(spec.scoped_context)
    ordering = [fillers[0], core[0], spec.surface_anchor, core[1], fillers[1], core[2]]
    if rng.random() < 0.5:
        ordering[0], ordering[1] = ordering[1], ordering[0]
    return "\n".join(f"Session {i+1}: {text}" for i, text in enumerate(ordering))


def _make_row(
    spec: ScenarioSpec,
    group_id: str,
    profile_variant: str,
    query_kind: str,
    query: str,
    history: str,
) -> Dict[str, str]:
    if profile_variant == "stable":
        action = "APPLY"
        zone = "inside"
        profile_type = "trait_preference"
        query_relation = "match" if query_kind == "shared" else ("mismatch" if query_kind == "crossing" else "unknown")
        applies_when = "broadly across relevant situations"
        not_when = "only when explicitly overridden by the user"
        temporal_validity = "persistent"
        owner = "user"
    else:
        action = {"shared": "APPLY", "crossing": "IGNORE", "uncertain": "CLARIFY"}[query_kind]
        zone = {"shared": "inside", "crossing": "outside", "uncertain": "near"}[query_kind]
        profile_type = spec.scoped_type
        query_relation = {"shared": "match", "crossing": "mismatch", "uncertain": "unknown"}[query_kind]
        applies_when = spec.applies_when
        not_when = spec.not_when
        temporal_validity = "temporary" if spec.scoped_type == "temporary_constraint" else "context-dependent"
        owner = "other" if spec.scoped_type == "other_owner" else "user"

    return {
        "group_id": group_id,
        "twin_pair_id": f"{group_id}:{query_kind}",
        "family": spec.family,
        "domain": spec.domain,
        "profile_variant": profile_variant,
        "profile_type": profile_type,
        "preference_label": spec.preference_label,
        "owner": owner,
        "applies_when": applies_when,
        "not_when": not_when,
        "temporal_validity": temporal_validity,
        "history": history,
        "query": query,
        "query_kind": query_kind,
        "query_relation": query_relation,
        "zone": zone,
        "action": action,
        "clarification_question": spec.clarification_question,
        "apply_response": spec.apply_response,
        "neutral_response": spec.neutral_response,
    }


def generate_frontier_suite(groups_per_family: int = 32, seed: int = 42) -> pd.DataFrame:
    """Generate a controlled counterfactual evaluation suite.

    Each group contains two histories with the same surface preference request:
    one encodes a broad stable preference and the other a scoped/temporary cause.
    The same three queries are asked against both histories.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for spec in SPECS:
        for idx in range(groups_per_family):
            group_id = f"{spec.family}-{idx:03d}"
            histories = {
                "stable": _history(spec, "stable", rng),
                "scoped": _history(spec, "scoped", rng),
            }
            queries = {
                "shared": rng.choice(spec.shared_queries),
                "crossing": rng.choice(spec.crossing_queries),
                "uncertain": rng.choice(spec.uncertain_queries),
            }
            for variant in ("stable", "scoped"):
                for query_kind, query in queries.items():
                    rows.append(
                        _make_row(
                            spec=spec,
                            group_id=group_id,
                            profile_variant=variant,
                            query_kind=query_kind,
                            query=query,
                            history=histories[variant],
                        )
                    )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_frontier_suite(path: str, groups_per_family: int = 32, seed: int = 42) -> pd.DataFrame:
    df = generate_frontier_suite(groups_per_family=groups_per_family, seed=seed)
    df.to_csv(path, index=False)
    return df
