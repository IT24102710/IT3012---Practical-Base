# test_logic.py
# Practical 05 - IT3012: Intelligent Agents
# Part 4: Self-Evaluation Test Cases for the Forward Chaining Engine

from logic_engine import KnowledgeBase


def test_forward_chaining():
    kb = KnowledgeBase()

    # Add Domain Rules
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    # ---------------------------------------------------------------
    # Test Case 1: Safe Engagement
    #   Facts : TargetVisible, HasDust
    #   Expect: SafeToEngage deduced, Retreat NOT deduced
    # ---------------------------------------------------------------
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()
    assert 'SafeToEngage' in kb.facts, "Test 1 Failed: Should deduce SafeToEngage"
    assert 'Retreat' not in kb.facts, "Test 1 Failed: Should NOT deduce Retreat"

    # ---------------------------------------------------------------
    # Test Case 2: Unsafe Engagement (Bloodseeker Missing)
    #   Facts : TargetVisible, HasDust, BloodseekerMissing
    #   Expect: Retreat deduced (via SafeToEngage → Retreat chain)
    # ---------------------------------------------------------------
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()
    assert 'Retreat' in kb.facts, "Test 2 Failed: Should deduce Retreat to override search"

    print("[PASS] All Logic Engine Test Cases Passed!")


if __name__ == "__main__":
    test_forward_chaining()
