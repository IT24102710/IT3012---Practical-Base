
class KnowledgeBase:
    """
    A propositional Knowledge Base that stores facts and Horn Clause rules,
    and can derive new facts via Forward Chaining (data-driven inference).
    """

    def __init__(self):
        
        self.facts = set()

        self.rules = []

    # ------------------------------------------------------------------
    # Part 1 - Knowledge Base API
    # ------------------------------------------------------------------

    def tell_fact(self, fact_string: str):
        """Assert a new fact into the Knowledge Base."""
        self.facts.add(fact_string)

    def tell_rule(self, premise_list: list, conclusion_string: str):
        """
        Register a Horn Clause rule.
        premise_list  – list of fact strings that must ALL be true
        conclusion_string – fact that is derived when all premises hold
        """
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        """Remove all current facts (reset percepts between tile evaluations)."""
        self.facts = set()

    # ------------------------------------------------------------------
    # Part 2 - Forward Chaining Inference Engine
    # ------------------------------------------------------------------

    def forward_chain(self):
        """
        Data-driven Forward Chaining using Modus Ponens.

        Iterates over all rules in a loop until a complete pass adds no
        new facts.  For every rule whose premises are all satisfied, the
        conclusion is added to the fact set.
        """
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    if all(p in self.facts for p in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True
