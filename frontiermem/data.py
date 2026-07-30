from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    family: str
    domain: str
    preference_label: str
    surface_anchors: Sequence[str]
    stable_context: Sequence[str]
    scoped_context: Sequence[str]
    scoped_type: str
    applies_when: str
    does_not_apply_when: str
    train_inside_queries: Sequence[str]
    test_inside_queries: Sequence[str]
    train_outside_queries: Sequence[str]
    test_outside_queries: Sequence[str]
    train_near_queries: Sequence[str]
    test_near_queries: Sequence[str]
    clarification_question: str
    apply_response: str
    neutral_response: str
    owner: str = "user"


FILLERS = (
    "We also discussed a calendar reminder for next month.",
    "Another short conversation concerned a book recommendation.",
    "There was a separate exchange about organizing files on a laptop.",
    "We briefly talked about a grocery list for the weekend.",
    "A different session was about fixing a small Python error.",
    "We also discussed the weather for an unrelated day trip.",
    "Another message asked for a summary of a news article.",
    "There was an unrelated request to rename several documents.",
)


SPECS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        family="writing_conciseness",
        domain="writing",
        preference_label="concise answers",
        surface_anchors=(
            "Please keep this answer short and direct.",
            "For this request, give me only the key points.",
        ),
        stable_context=(
            "I generally prefer concise answers across most tasks.",
            "Even when I have time, I usually want the direct answer first.",
            "Brevity is a persistent interaction preference for me.",
            "Across our conversations, short responses are my default.",
        ),
        scoped_context=(
            "I am about to join a meeting, so this request is unusually urgent.",
            "The short answer is only for time-pressured situations.",
            "When learning something new, I normally want a detailed explanation.",
            "This is not a general preference for every conversation.",
        ),
        scoped_type="temporary_state",
        applies_when="the request is urgent, exam-like, or explicitly asks for a direct answer",
        does_not_apply_when="the user is learning a new concept or preparing a detailed presentation",
        train_inside_queries=(
            "I have two minutes before an exam; just give the answer.",
            "This is urgent, so reply with only the conclusion.",
            "Give me a direct answer without the derivation this time.",
        ),
        test_inside_queries=(
            "I am in a hurry; provide only the final result.",
            "Treat this as a rapid-fire question and keep it brief.",
        ),
        train_outside_queries=(
            "Teach me this unfamiliar topic from the beginning with full reasoning.",
            "I am preparing a research talk and need a thorough explanation.",
            "Walk me through the concept carefully with examples.",
        ),
        test_outside_queries=(
            "I want to understand this deeply, so explain each step.",
            "This is for learning, not an exam; give me the detailed version.",
        ),
        train_near_queries=(
            "Explain this to me, but I have not decided how much detail I want.",
            "I need an answer, although the desired level of detail is unclear.",
            "Respond in the style I need, but I have not specified the context.",
        ),
        test_near_queries=(
            "Should this be a quick answer or a tutorial? I have not decided.",
            "I am unsure whether I need the summary or the full explanation.",
        ),
        clarification_question="Would you like only the final answer, or a detailed step-by-step explanation?",
        apply_response="I will keep the answer concise and give the main result first.",
        neutral_response="I will not force the earlier brevity preference onto this learning request.",
    ),
    ScenarioSpec(
        family="writing_formality",
        domain="writing",
        preference_label="formal writing style",
        surface_anchors=(
            "Please rewrite this message in a formal tone.",
            "Make this draft professional and formal.",
        ),
        stable_context=(
            "I consistently prefer formal wording in written communication.",
            "Professional language is my default even with familiar colleagues.",
            "Across audiences, I generally avoid casual expressions.",
            "A formal tone is a stable writing preference for me.",
        ),
        scoped_context=(
            "This draft is addressed to a professor, so the audience requires formality.",
            "I use a relaxed tone with friends and close teammates.",
            "The formal request applies only to academic or official recipients.",
            "It is an audience-specific requirement, not my universal style.",
        ),
        scoped_type="contextual_preference",
        applies_when="the recipient is a professor, administrator, employer, or other official audience",
        does_not_apply_when="the recipient is a close friend or an informal peer group",
        train_inside_queries=(
            "Rewrite this email to the department chair.",
            "Prepare a message for a scholarship committee.",
            "Polish this note before I send it to my supervisor.",
        ),
        test_inside_queries=(
            "Draft a reply to the university administration.",
            "Help me write to a potential employer.",
        ),
        train_outside_queries=(
            "Write a funny message to my closest friend.",
            "Turn this into a casual text for my roommates.",
            "Help me reply informally in our gaming group.",
        ),
        test_outside_queries=(
            "Make this sound natural for a group chat with friends.",
            "Write a playful note to my cousin.",
        ),
        train_near_queries=(
            "Rewrite this message, but I have not said who will receive it.",
            "Improve the tone; the audience is still unknown.",
            "Make this message appropriate, although I have not identified the recipient.",
        ),
        test_near_queries=(
            "Polish this text; I am not sure yet whether it is for a professor or a friend.",
            "Adjust the tone, but the recipient has not been decided.",
        ),
        clarification_question="Who is the recipient: an official contact or someone you know casually?",
        apply_response="I will use a formal, professional tone suitable for the recipient.",
        neutral_response="I will use a natural informal tone rather than transferring the official style.",
    ),
    ScenarioSpec(
        family="writing_directness",
        domain="writing",
        preference_label="direct communication",
        surface_anchors=(
            "Please make the message very direct.",
            "Remove hedging and state the request clearly.",
        ),
        stable_context=(
            "I generally value direct communication, even in sensitive discussions.",
            "Across contexts, I prefer people to state the point clearly.",
            "Avoiding unnecessary hedging is a persistent preference of mine.",
            "Direct wording is my default style.",
        ),
        scoped_context=(
            "This is an internal logistics message where speed matters.",
            "For emotional or conflict-sensitive conversations, I prefer gentler wording.",
            "The direct style is limited to operational requests.",
            "It is not my general approach for every audience.",
        ),
        scoped_type="contextual_preference",
        applies_when="the message concerns logistics, deadlines, or clear task assignment",
        does_not_apply_when="the message addresses grief, conflict, rejection, or another sensitive situation",
        train_inside_queries=(
            "Tell the team that the file is due by noon.",
            "Assign the remaining tasks and state the deadline.",
            "Write a clear request for the missing document.",
        ),
        test_inside_queries=(
            "Send a straightforward reminder about tomorrow's submission.",
            "Ask the contractor to fix the issue by Friday.",
        ),
        train_outside_queries=(
            "Help me comfort a friend who is grieving.",
            "Write a tactful rejection after a difficult interview.",
            "Respond gently to a teammate who feels hurt.",
        ),
        test_outside_queries=(
            "Help me apologize after an emotional argument.",
            "Write a compassionate message to someone facing bad news.",
        ),
        train_near_queries=(
            "Draft a message, but I have not explained whether the topic is sensitive.",
            "Make this appropriate; the emotional stakes are unclear.",
            "I need a reply, but I have not described the relationship or situation.",
        ),
        test_near_queries=(
            "Write the message, although I have not said whether this is a conflict.",
            "Choose the level of directness, but the context is still ambiguous.",
        ),
        clarification_question="Is this a routine logistical request, or a sensitive interpersonal message?",
        apply_response="I will state the request directly and remove unnecessary hedging.",
        neutral_response="I will use gentler wording because the earlier direct style does not fit this context.",
    ),
    ScenarioSpec(
        family="education_detail",
        domain="education",
        preference_label="detailed step-by-step explanations",
        surface_anchors=(
            "Please explain this step by step.",
            "I want a detailed explanation with the intermediate reasoning.",
        ),
        stable_context=(
            "I consistently learn best from detailed step-by-step explanations.",
            "Even for familiar material, I prefer seeing the reasoning process.",
            "Detailed teaching is a stable learning preference for me.",
            "Across subjects, examples and intermediate steps help me most.",
        ),
        scoped_context=(
            "This topic is new to me, so I need extra detail for this lesson.",
            "For routine checks and timed quizzes, I usually want only the answer.",
            "The detailed request is limited to unfamiliar learning material.",
            "It is not a universal response-style preference.",
        ),
        scoped_type="current_goal",
        applies_when="the user is learning unfamiliar material or explicitly studying for understanding",
        does_not_apply_when="the user is checking a known result or taking a timed quiz",
        train_inside_queries=(
            "Teach me this concept; I have never studied it before.",
            "I am learning this chapter and need the full derivation.",
            "Explain every step because the method is unfamiliar.",
        ),
        test_inside_queries=(
            "I am new to this subject, so build the explanation from first principles.",
            "Help me truly learn the method with all intermediate steps.",
        ),
        train_outside_queries=(
            "I know the method; just verify my final answer.",
            "This is a timed quiz, so give only the correct option.",
            "Check whether my result is right without reteaching the topic.",
        ),
        test_outside_queries=(
            "I only need a quick correctness check on a familiar exercise.",
            "For this speed test, return the final answer only.",
        ),
        train_near_queries=(
            "Answer this question, but I have not said whether I am learning or checking.",
            "I need help with the problem; my familiarity is unknown.",
            "Explain it appropriately, although my goal is not specified.",
        ),
        test_near_queries=(
            "I need assistance, but I have not said whether this is study or verification.",
            "Choose the amount of explanation; my current goal is unclear.",
        ),
        clarification_question="Are you learning this from scratch, or do you only want the final answer checked?",
        apply_response="I will explain the method carefully and include the intermediate steps.",
        neutral_response="I will provide a concise verification rather than a full tutorial.",
    ),
    ScenarioSpec(
        family="education_examples",
        domain="education",
        preference_label="example-driven teaching",
        surface_anchors=(
            "Please teach this with several concrete examples.",
            "Use examples rather than only abstract definitions.",
        ),
        stable_context=(
            "I generally understand ideas best through concrete examples.",
            "Across technical subjects, examples are my preferred learning method.",
            "I consistently ask for at least one worked example.",
            "Example-driven teaching is a stable preference for me.",
        ),
        scoped_context=(
            "This concept is unusually abstract, so examples are needed for this topic.",
            "For reference sheets, I prefer compact definitions without examples.",
            "The request for examples applies only to abstract learning sessions.",
            "It is not required in every educational response.",
        ),
        scoped_type="contextual_preference",
        applies_when="the concept is abstract, unfamiliar, or explicitly taught for understanding",
        does_not_apply_when="the user requests a compact reference, formula sheet, or glossary entry",
        train_inside_queries=(
            "Explain this abstract idea using a real-world example.",
            "Teach the concept with a worked numerical case.",
            "I cannot visualize this; give me concrete examples.",
        ),
        test_inside_queries=(
            "Use an illustrative case to make this unfamiliar idea clear.",
            "Show how the concept works in a practical example.",
        ),
        train_outside_queries=(
            "Create a one-page formula sheet with no examples.",
            "Give me a compact glossary definition.",
            "List the rules only; I am making a reference card.",
        ),
        test_outside_queries=(
            "Produce a terse cheat sheet containing definitions only.",
            "I need a compact reference entry, not a worked example.",
        ),
        train_near_queries=(
            "Explain this, but I have not said whether it is for learning or reference.",
            "Prepare the material; the intended format is unclear.",
            "I need a description, although I have not specified whether examples are useful.",
        ),
        test_near_queries=(
            "Help with this concept, but the final use of the answer is unknown.",
            "Choose whether to include examples; I have not stated my purpose.",
        ),
        clarification_question="Is this for learning the idea, or for a compact reference sheet?",
        apply_response="I will include concrete examples and connect them to the abstract idea.",
        neutral_response="I will keep this as a compact reference without adding worked examples.",
    ),
    ScenarioSpec(
        family="education_answer_only",
        domain="education",
        preference_label="answer-only assistance",
        surface_anchors=(
            "Just tell me the correct answer.",
            "For this question, do not include the derivation.",
        ),
        stable_context=(
            "I consistently prefer answer-only assistance for exercises.",
            "Across subjects, I usually work through the reasoning independently.",
            "The final result is my persistent preference when asking for help.",
            "Even without time pressure, I want only the answer.",
        ),
        scoped_context=(
            "This is a timed practice round, so I need only the answer right now.",
            "Outside timed practice, I normally want explanations and hints.",
            "The answer-only request is limited to speed drills.",
            "It is a temporary task constraint, not my general learning style.",
        ),
        scoped_type="temporary_constraint",
        applies_when="the user is in a timed drill, speed test, or explicitly requests answer-only output",
        does_not_apply_when="the user is studying, reviewing mistakes, or asks for conceptual understanding",
        train_inside_queries=(
            "This is a sixty-second drill; return only the option.",
            "I am timing myself, so give the answer without explanation.",
            "For this speed round, output only the result.",
        ),
        test_inside_queries=(
            "I have almost no time; state only the correct choice.",
            "Treat this as a rapid quiz and skip the reasoning.",
        ),
        train_outside_queries=(
            "Review my mistake and explain why the correct answer works.",
            "I am studying this topic and want to understand the reasoning.",
            "Teach me how to solve similar questions next time.",
        ),
        test_outside_queries=(
            "Help me learn from this error with a full explanation.",
            "I am no longer timing myself; show me the reasoning.",
        ),
        train_near_queries=(
            "Help with this question, but I have not said whether it is timed.",
            "Give me the appropriate response; my purpose is unclear.",
            "I need assistance, although I have not specified drill or study mode.",
        ),
        test_near_queries=(
            "Respond to the problem, but I have not described the learning context.",
            "I am unsure whether I want speed or explanation for this one.",
        ),
        clarification_question="Is this a timed drill where you want only the answer, or a study question requiring explanation?",
        apply_response="I will provide only the final answer.",
        neutral_response="I will explain the reasoning because the answer-only constraint does not apply here.",
    ),
    ScenarioSpec(
        family="travel_budget",
        domain="travel",
        preference_label="budget-conscious accommodation",
        surface_anchors=(
            "For this trip, please find a cheap but clean hotel.",
            "Recommend an inexpensive hotel that is still acceptable.",
        ),
        stable_context=(
            "I generally choose the least expensive clean hotel, even when someone else pays.",
            "Across my trips, I care more about price than luxury.",
            "Saving money on accommodation is a persistent priority for me.",
            "Budget hotels are my default travel preference.",
        ),
        scoped_context=(
            "I usually enjoy comfortable premium hotels.",
            "This trip is different because the university reimbursement cap is strict.",
            "The low-price request is only for this capped trip.",
            "Cheap accommodation is not a general travel preference for me.",
        ),
        scoped_type="temporary_constraint",
        applies_when="the traveler is paying personally or a strict reimbursement cap applies",
        does_not_apply_when="the trip is fully reimbursed without a cap",
        train_inside_queries=(
            "I am paying for this trip myself and my budget is tight.",
            "For a self-funded weekend with a strict budget, recommend a hotel.",
            "I need accommodation that stays within my limited personal budget.",
        ),
        test_inside_queries=(
            "This journey comes out of my own pocket, so cost matters a lot.",
            "Find lodging for a trip with a firm reimbursement ceiling.",
        ),
        train_outside_queries=(
            "My employer will cover the full hotel cost with no reimbursement cap.",
            "This conference hotel is fully paid by the company.",
            "The university confirmed unlimited hotel reimbursement.",
        ),
        test_outside_queries=(
            "The organization is paying and there is no spending limit for lodging.",
            "Accommodation is completely reimbursed, without a price ceiling.",
        ),
        train_near_queries=(
            "I do not know the reimbursement details yet.",
            "The payment policy is still unclear.",
            "I have not checked who will pay for the hotel.",
        ),
        test_near_queries=(
            "I need a hotel, but the funding arrangement has not been decided.",
            "It is unclear whether I or the sponsor will cover accommodation.",
        ),
        clarification_question="Will you pay personally, or is this trip reimbursed without a cap?",
        apply_response="I will prioritize inexpensive, clean accommodation.",
        neutral_response="I will consider comfort and quality rather than assuming price is the main criterion.",
    ),
    ScenarioSpec(
        family="travel_proximity",
        domain="travel",
        preference_label="accommodation close to the destination",
        surface_anchors=(
            "Find a hotel as close to the venue as possible.",
            "For this trip, prioritize walking distance to the destination.",
        ),
        stable_context=(
            "I generally prioritize location over hotel amenities.",
            "Across trips, staying close to the main destination is my default.",
            "Even on relaxed vacations, I dislike long travel from the hotel.",
            "Proximity is a persistent accommodation preference for me.",
        ),
        scoped_context=(
            "This is a one-day business visit with meetings from early morning.",
            "The tight schedule makes proximity important only for this trip.",
            "On normal vacations, I prefer quieter hotels outside crowded centers.",
            "The walking-distance request is not a general preference.",
        ),
        scoped_type="current_goal",
        applies_when="the trip is a short business visit with a tightly packed schedule",
        does_not_apply_when="the trip is a relaxed multi-day leisure vacation",
        train_inside_queries=(
            "I have back-to-back meetings and only one night in the city.",
            "This is a short work visit with an early event at the venue.",
            "I need to walk to several meetings during a packed day.",
        ),
        test_inside_queries=(
            "The schedule is compressed and I must reach the conference hall before dawn.",
            "I am visiting for one busy workday with no time for commuting.",
        ),
        train_outside_queries=(
            "This is a relaxed week-long holiday and I want a quiet neighborhood.",
            "I have many free days and prefer staying outside the tourist center.",
            "For this leisurely vacation, a peaceful area matters more than proximity.",
        ),
        test_outside_queries=(
            "The trip is an unhurried retreat, so I would rather stay away from downtown.",
            "I have a long holiday and do not mind traveling to attractions.",
        ),
        train_near_queries=(
            "I need a hotel, but I have not described the trip schedule.",
            "Recommend accommodation; the length and pace of the trip are unknown.",
            "I have not said whether this visit is business or leisure.",
        ),
        test_near_queries=(
            "Suggest lodging, although I have not decided the trip itinerary.",
            "The purpose and schedule of the visit are still unclear.",
        ),
        clarification_question="Is this a tightly scheduled short visit, or a relaxed leisure trip?",
        apply_response="I will prioritize hotels close to the main destination.",
        neutral_response="I will consider quieter or more comfortable areas instead of assuming proximity is essential.",
    ),
    ScenarioSpec(
        family="travel_comfort",
        domain="travel",
        preference_label="high-comfort travel",
        surface_anchors=(
            "For this trip, choose the most comfortable option.",
            "Prioritize comfort over saving a small amount of money.",
        ),
        stable_context=(
            "I generally prioritize comfort on every trip.",
            "Even for short journeys, I avoid inconvenient travel options.",
            "Comfort is a persistent travel preference for me.",
            "Across trips, I am willing to pay more for an easier experience.",
        ),
        scoped_context=(
            "I am recovering from a temporary back injury.",
            "Extra comfort is medically useful during this recovery period.",
            "After recovery, I usually accept basic low-cost travel.",
            "This comfort requirement is temporary, not a universal preference.",
        ),
        scoped_type="temporary_state",
        applies_when="the user is currently injured, physically exhausted, or needs accessibility support",
        does_not_apply_when="the user has recovered and is planning a routine low-cost trip",
        train_inside_queries=(
            "My back is still painful, so I need an easy journey.",
            "I am currently recovering and cannot tolerate a difficult itinerary.",
            "Because of the injury, prioritize comfort and accessibility.",
        ),
        test_inside_queries=(
            "I am not fully healed yet and need the least strenuous option.",
            "My current physical condition requires extra travel comfort.",
        ),
        train_outside_queries=(
            "I have fully recovered and this is a routine budget trip.",
            "The injury is gone, and I want the cheapest reasonable route.",
            "I feel healthy again and do not need premium comfort.",
        ),
        test_outside_queries=(
            "Recovery is complete, so basic transportation is fine now.",
            "I am healthy and planning an ordinary low-cost journey.",
        ),
        train_near_queries=(
            "Plan the trip, but I have not said whether the injury has healed.",
            "I need transportation; my current physical condition is unknown.",
            "Choose an option, although I have not updated you about recovery.",
        ),
        test_near_queries=(
            "Recommend travel, but it is unclear whether I still need accessibility support.",
            "My present recovery status has not been specified.",
        ),
        clarification_question="Do you still need extra comfort or accessibility support for this trip?",
        apply_response="I will prioritize comfort, accessibility, and a low-strain itinerary.",
        neutral_response="I will not assume the temporary comfort requirement still applies.",
    ),
    ScenarioSpec(
        family="food_vegetarian",
        domain="food",
        preference_label="vegetarian meals",
        surface_anchors=(
            "Please suggest a vegetarian meal.",
            "For this request, avoid meat and fish.",
        ),
        stable_context=(
            "I am consistently vegetarian in my own meals.",
            "Across restaurants and home cooking, I avoid meat and fish.",
            "Vegetarian eating is a persistent personal preference.",
            "Even at celebrations, I choose vegetarian dishes.",
        ),
        scoped_context=(
            "The vegetarian guest is my sister, not me.",
            "I normally eat meat and fish myself.",
            "This meat-free request applies only when I am cooking for her.",
            "It should not be stored as my own general diet.",
        ),
        scoped_type="other_owner",
        applies_when="the meal is for the vegetarian sister or another explicitly vegetarian guest",
        does_not_apply_when="the meal is for the user alone and no vegetarian guest is involved",
        train_inside_queries=(
            "My vegetarian sister is joining dinner tonight.",
            "I am cooking for the same guest who avoids meat.",
            "Recommend a dish for my sister's vegetarian birthday meal.",
        ),
        test_inside_queries=(
            "The meat-free guest will be eating with us again.",
            "Plan dinner for my sister, who still follows a vegetarian diet.",
        ),
        train_outside_queries=(
            "I am cooking only for myself tonight and I eat meat.",
            "My sister is not coming; suggest a meal just for me.",
            "No vegetarian guests are involved in this dinner.",
        ),
        test_outside_queries=(
            "This meal is solely for me, and I have no meat restriction.",
            "I am dining alone today; the vegetarian guest is absent.",
        ),
        train_near_queries=(
            "Suggest dinner, but I have not said who will eat it.",
            "Plan a meal; the guests are still unknown.",
            "I need a recipe, although I have not identified the diner.",
        ),
        test_near_queries=(
            "Recommend food, but I have not confirmed whether my sister is attending.",
            "The intended diner has not been specified yet.",
        ),
        clarification_question="Is this meal for you, or for the vegetarian guest you mentioned?",
        apply_response="I will recommend a fully vegetarian meal.",
        neutral_response="I will not treat another person's diet as your own preference.",
        owner="sister",
    ),
    ScenarioSpec(
        family="food_spiciness",
        domain="food",
        preference_label="mild food",
        surface_anchors=(
            "Please make the dish mild rather than spicy.",
            "For this meal, avoid strong chili heat.",
        ),
        stable_context=(
            "I generally dislike spicy food.",
            "Across cuisines, I consistently choose mild dishes.",
            "Avoiding chili heat is a persistent preference for me.",
            "Even when others order spicy food, I select the mild option.",
        ),
        scoped_context=(
            "My stomach is temporarily sensitive because of medication.",
            "Normally I enjoy very spicy food.",
            "The mild-food request applies only during this treatment period.",
            "It is a temporary health constraint, not my usual taste.",
        ),
        scoped_type="temporary_state",
        applies_when="the user is still taking the medication or has an active stomach problem",
        does_not_apply_when="the treatment has ended and the stomach problem has resolved",
        train_inside_queries=(
            "I am still taking the medication and my stomach is sensitive.",
            "The treatment continues, so recommend something gentle.",
            "My stomach symptoms are active today.",
        ),
        test_inside_queries=(
            "I remain on the medicine and cannot handle strong chili yet.",
            "The stomach issue has not resolved, so keep the meal gentle.",
        ),
        train_outside_queries=(
            "The treatment ended and I want my usual spicy food again.",
            "My stomach has recovered; recommend a hot chili dish.",
            "I am healthy now and would like strong spice.",
        ),
        test_outside_queries=(
            "The medication is finished and the sensitivity is gone.",
            "I have recovered and want a dish with plenty of heat.",
        ),
        train_near_queries=(
            "Suggest food, but I have not said whether I am still on the medication.",
            "I need dinner; my current stomach condition is unclear.",
            "Recommend a dish, although I have not updated you about treatment.",
        ),
        test_near_queries=(
            "Plan a meal, but it is unknown whether the sensitivity remains.",
            "I have not said whether the medicine course is complete.",
        ),
        clarification_question="Is your stomach still sensitive, or has the temporary restriction ended?",
        apply_response="I will recommend a mild dish with little or no chili heat.",
        neutral_response="I will not assume the temporary mild-food restriction still applies.",
    ),
    ScenarioSpec(
        family="food_quick_meals",
        domain="food",
        preference_label="quick-to-prepare meals",
        surface_anchors=(
            "Suggest a meal that takes very little time to prepare.",
            "For this request, prioritize speed and convenience.",
        ),
        stable_context=(
            "I generally prefer quick meals because I dislike lengthy cooking.",
            "Across weekdays and weekends, preparation time is a major priority.",
            "Fast recipes are my persistent cooking preference.",
            "Even when I am free, I usually choose simple meals.",
        ),
        scoped_context=(
            "This week I have an unusually heavy deadline.",
            "Normally I enjoy slow cooking when my schedule is open.",
            "The quick-meal request applies only during the deadline period.",
            "It is a temporary time constraint, not my general cooking preference.",
        ),
        scoped_type="current_goal",
        applies_when="the user is currently facing a deadline or has very limited cooking time",
        does_not_apply_when="the deadline has passed and the user has time for relaxed cooking",
        train_inside_queries=(
            "The deadline is tomorrow, so I have only fifteen minutes to cook.",
            "I am still overloaded with work and need a fast dinner.",
            "My schedule is packed tonight; suggest something quick.",
        ),
        test_inside_queries=(
            "I am racing to finish a project and need food immediately.",
            "There is almost no cooking time before tonight's deadline.",
        ),
        train_outside_queries=(
            "The deadline is over and I want a slow weekend cooking project.",
            "I am free today and would enjoy a complex recipe.",
            "My schedule is open, so preparation time is not a concern.",
        ),
        test_outside_queries=(
            "Work is finished and I have the whole afternoon to cook.",
            "I am relaxed this weekend and want a time-intensive dish.",
        ),
        train_near_queries=(
            "Suggest a meal, but I have not described today's schedule.",
            "I need a recipe; my available cooking time is unknown.",
            "Recommend dinner, although I have not said whether the deadline remains.",
        ),
        test_near_queries=(
            "Plan food for today, but my time constraints are unclear.",
            "I have not said whether I am still busy or finally free.",
        ),
        clarification_question="Are you still under time pressure, or do you have time for a longer recipe?",
        apply_response="I will prioritize a fast, low-preparation meal.",
        neutral_response="I will consider a more involved recipe instead of assuming you are still time-constrained.",
    ),
)


SPEC_BY_FAMILY = {spec.family: spec for spec in SPECS}


def _sample_history(spec: ScenarioSpec, profile_variant: str, rng: Random) -> str:
    core = list(spec.stable_context if profile_variant == "stable" else spec.scoped_context)
    rng.shuffle(core)
    anchor = rng.choice(tuple(spec.surface_anchors))
    fillers = rng.sample(tuple(FILLERS), k=2)
    messages = [core[0], fillers[0], anchor, core[1], fillers[1], core[2], core[3]]
    # Shuffle only the middle while ensuring the anchor and evidence remain visible.
    middle = messages[1:-1]
    rng.shuffle(middle)
    messages = [messages[0], *middle, messages[-1]]
    return "\n".join(f"Session {i + 1}: {text}" for i, text in enumerate(messages))


def _gold_action(profile_variant: str, query_kind: str) -> tuple[str, str]:
    if profile_variant == "stable":
        return "APPLY", "inside"
    if query_kind == "inside":
        return "APPLY", "inside"
    if query_kind == "outside":
        return "IGNORE", "outside"
    return "CLARIFY", "near"


def _query_pool(spec: ScenarioSpec, split: str, kind: str) -> Sequence[str]:
    attr = f"{split}_{kind}_queries"
    return getattr(spec, attr)


def generate_frontier_suite(
    train_groups_per_family: int = 48,
    test_groups_per_family: int = 16,
    seed: int = 17,
) -> pd.DataFrame:
    """Generate controlled counterfactual twins with a strict group split.

    Training groups only use train paraphrases; test groups only use held-out paraphrases.
    Each group contains a stable and scoped history, each queried in inside/outside/near regimes.
    """
    rows: list[dict] = []
    rng = Random(seed)
    global_group = 0

    for spec in SPECS:
        for split, group_count in (("train", train_groups_per_family), ("test", test_groups_per_family)):
            for local_group in range(group_count):
                global_group += 1
                group_id = f"{spec.family}-{split}-{local_group:03d}"
                stable_history = _sample_history(spec, "stable", rng)
                scoped_history = _sample_history(spec, "scoped", rng)
                for query_kind in ("inside", "outside", "near"):
                    query = rng.choice(tuple(_query_pool(spec, split, query_kind)))
                    twin_pair_id = f"{group_id}-{query_kind}"
                    for profile_variant, history in (
                        ("stable", stable_history),
                        ("scoped", scoped_history),
                    ):
                        action, zone = _gold_action(profile_variant, query_kind)
                        information_type = (
                            "trait_preference" if profile_variant == "stable" else spec.scoped_type
                        )
                        owner = "user" if profile_variant == "stable" else spec.owner
                        temporal_validity = (
                            "persistent"
                            if profile_variant == "stable"
                            else "context-dependent"
                        )
                        rows.append(
                            {
                                "example_id": f"{twin_pair_id}-{profile_variant}",
                                "group_id": group_id,
                                "twin_pair_id": twin_pair_id,
                                "split": split,
                                "family": spec.family,
                                "domain": spec.domain,
                                "profile_variant": profile_variant,
                                "query_kind": query_kind,
                                "zone": zone,
                                "history": history,
                                "query": query,
                                "action": action,
                                "preference_label": spec.preference_label,
                                "owner": owner,
                                "information_type": information_type,
                                "temporal_validity": temporal_validity,
                                "applies_when": (
                                    "broadly across relevant situations"
                                    if profile_variant == "stable"
                                    else spec.applies_when
                                ),
                                "does_not_apply_when": (
                                    "only when explicitly overridden or corrected"
                                    if profile_variant == "stable"
                                    else spec.does_not_apply_when
                                ),
                                "clarification_question": spec.clarification_question,
                                "apply_response": spec.apply_response,
                                "neutral_response": spec.neutral_response,
                            }
                        )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def save_frontier_suite(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def scenario_catalog() -> list[dict]:
    return [asdict(spec) for spec in SPECS]


__all__ = [
    "ScenarioSpec",
    "SPECS",
    "SPEC_BY_FAMILY",
    "generate_frontier_suite",
    "save_frontier_suite",
    "scenario_catalog",
]
