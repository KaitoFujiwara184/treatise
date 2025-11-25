from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List

CARDS = ("T", "J", "Q", "K", "A")
CARD_STRENGTH = {card: rank for rank, card in enumerate(CARDS)}
NUM_ROUNDS = 3
ANTE = 1


def card_is_stronger(card_a: str, card_b: str) -> bool:
    return CARD_STRENGTH[card_a] > CARD_STRENGTH[card_b]


@dataclass
class Node:
    info_set: str
    regret_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])
    strategy_sum: List[float] = field(default_factory=lambda: [0.0, 0.0])

    def get_strategy(self, reach_probability: float) -> List[float]:
        positive_regrets = [max(r, 0.0) for r in self.regret_sum]
        normalizing_sum = sum(positive_regrets)
        if normalizing_sum > 0:
            strategy = [r / normalizing_sum for r in positive_regrets]
        else:
            strategy = [0.5, 0.5]
        self.strategy_sum = [s + reach_probability * a for s, a in zip(self.strategy_sum, strategy)]
        return strategy

    def get_average_strategy(self) -> List[float]:
        normalizing_sum = sum(self.strategy_sum)
        if normalizing_sum > 0:
            return [s / normalizing_sum for s in self.strategy_sum]
        return [0.5, 0.5]


class KuhnTrainer:
    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}

    def train(self, iterations: int, *, progress: bool = True) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        cards = list(CARDS)
        util = 0.0
        report_every = max(iterations // 20, 1)
        for i in range(iterations):
            random.shuffle(cards)
            traverser = i % 2
            util += self.mccfr(
                cards,
                0,
                [],
                "",
                traverser,
                (float(ANTE), float(ANTE)),
                1.0,
                1.0,
            )
            if progress and (i + 1) % report_every == 0:
                print(f"Iteration {i + 1}/{iterations}: average game value = {util / (i + 1):.6f}")

    def mccfr(
        self,
        cards: List[str],
        round_index: int,
        completed_rounds: List[str],
        current_round_actions: str,
        traverser: int,
        contributions: tuple[float, float],
        p_traverser: float,
        p_opponent: float,
    ) -> float:

        terminal_utility = self._terminal_utility(
            current_round_actions,
            round_index,
            completed_rounds,
            contributions,
            cards,
            traverser,
        )
        if terminal_utility is not None:
            return terminal_utility

        if self._is_round_over(current_round_actions) and round_index < NUM_ROUNDS - 1:
            next_round, next_completed, next_actions = self._advance_round_if_needed(
                round_index, completed_rounds, current_round_actions
            )
            return self.mccfr(
                cards,
                next_round,
                next_completed,
                next_actions,
                traverser,
                contributions,
                p_traverser,
                p_opponent,
            )

        player = len(current_round_actions) % 2

        info_set_history = self._format_history(completed_rounds, current_round_actions)
        info_set = f"{cards[player]}:{info_set_history}"
        node = self.nodes.setdefault(info_set, Node(info_set))

        owner_reach = p_traverser if player == traverser else p_opponent
        strategy = node.get_strategy(owner_reach)

        if player == traverser:
            utils = [0.0, 0.0]
            node_util = 0.0
            for a, action in enumerate("pb"):
                next_actions = current_round_actions + action
                next_contribs = self._update_contributions(contributions, player, round_index, current_round_actions, action)
                next_state = self._advance_round_if_needed(round_index, completed_rounds, next_actions)
                next_p_traverser = p_traverser * strategy[a]
                utils[a] = self.mccfr(
                    cards,
                    next_state[0],
                    next_state[1],
                    next_state[2],
                    traverser,
                    next_contribs,
                    next_p_traverser,
                    p_opponent,
                )
                node_util += strategy[a] * utils[a]

            for a in range(2):
                regret = utils[a] - node_util
                node.regret_sum[a] += p_opponent * regret

            return node_util

        sampled_action = self._sample_action(strategy)
        next_actions = current_round_actions + sampled_action
        next_contribs = self._update_contributions(contributions, player, round_index, current_round_actions, sampled_action)
        next_state = self._advance_round_if_needed(round_index, completed_rounds, next_actions)
        next_p_opponent = p_opponent * strategy["pb".index(sampled_action)]
        return self.mccfr(
            cards,
            next_state[0],
            next_state[1],
            next_state[2],
            traverser,
            next_contribs,
            p_traverser,
            next_p_opponent,
        )

    def get_average_strategies(self) -> Dict[str, List[float]]:
        return {info_set: node.get_average_strategy() for info_set, node in self.nodes.items()}

    @staticmethod
    def _format_history(completed_rounds: List[str], current_round_actions: str) -> str:
        return "|".join(completed_rounds + [current_round_actions])

    def _advance_round_if_needed(
        self, round_index: int, completed_rounds: List[str], current_round_actions: str
    ) -> tuple[int, List[str], str]:
        if not self._is_round_over(current_round_actions):
            return round_index, completed_rounds, current_round_actions

        if self._fold_winner(current_round_actions) is not None:
            return round_index, completed_rounds, current_round_actions

        if round_index >= NUM_ROUNDS - 1:
            return round_index, completed_rounds, current_round_actions

        next_round = round_index + 1
        return next_round, completed_rounds + [current_round_actions], ""

    @staticmethod
    def _is_round_over(round_actions: str) -> bool:
        if len(round_actions) < 2:
            return False
        if round_actions[-1] == "p":
            return True
        return round_actions[-2:] == "bb"

    @staticmethod
    def _fold_winner(round_actions: str) -> int | None:
        if round_actions and round_actions[-1] == "p" and "b" in round_actions:
            bettor = round_actions.index("b") % 2
            return bettor
        return None

    def _terminal_utility(
        self,
        current_round_actions: str,
        round_index: int,
        completed_rounds: List[str],
        contributions: tuple[float, float],
        cards: List[str],
        player: int,
    ) -> float | None:
        if not self._is_round_over(current_round_actions):
            return None

        fold_winner = self._fold_winner(current_round_actions)
        if fold_winner is not None:
            return self._payoff(contributions, fold_winner, player)

        if round_index < NUM_ROUNDS - 1:
            return None

        winner = 0 if card_is_stronger(cards[0], cards[1]) else 1
        return self._payoff(contributions, winner, player)

    @staticmethod
    def _payoff(contributions: tuple[float, float], winner: int, player: int) -> float:
        pot = sum(contributions)
        payoff_p0 = pot - contributions[0] if winner == 0 else -contributions[0]
        return payoff_p0 if player == 0 else -payoff_p0

    @staticmethod
    def _update_contributions(
        contributions: tuple[float, float],
        player: int,
        round_index: int,
        round_actions: str,
        action: str,
    ) -> tuple[float, float]:
        if action != "b":
            return contributions
        if "b" in round_actions:
            outstanding = contributions[1 - player] - contributions[player]
            call_size = max(outstanding, 0.0)
            updated = list(contributions)
            updated[player] += call_size
            return updated[0], updated[1]
        pot_size = sum(contributions)
        bet_size = pot_size / 2.0
        updated = list(contributions)
        updated[player] += bet_size
        return updated[0], updated[1]

    @staticmethod
    def _sample_action(strategy: List[float]) -> str:
        r = random.random()
        return "p" if r < strategy[0] else "b"


def print_strategy_table(strategies: Dict[str, List[float]]) -> None:
    """Pretty-print a table of strategies keyed by information set."""
    print("Card ranking (weakest -> strongest): T < J < Q < K < A")
    header = "{:<8} | {:>6} | {:>6}".format("Info set", "Pass", "Bet")
    print(header)
    print("-" * len(header))
    history_order = {"": 0, "p": 1, "pb": 2, "b": 3, "bp": 4, "bb": 5}

    def sort_key(info_set: str) -> tuple[int, int, List[int], str]:
        card, history = info_set.split(":", maxsplit=1)
        per_round = history.split("|")
        round_orders = [history_order.get(h, len(h)) for h in per_round]
        return CARD_STRENGTH[card], len(per_round), round_orders, history

    for info_set in sorted(strategies, key=sort_key):
        strat = strategies[info_set]
        print("{:<8} | {:6.3f} | {:6.3f}".format(info_set, strat[0], strat[1]))



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=100_000_000,
        help="number of training iterations to run (default: 100000)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="suppress periodic progress updates",
    )
    args = parser.parse_args()

    iterations = args.iterations
    trainer = KuhnTrainer()
    trainer.train(iterations, progress=not args.no_progress)
    strategies = trainer.get_average_strategies()
    print("\nApproximate equilibrium strategies after", iterations, "iterations:")
    print_strategy_table(strategies)


if __name__ == "__main__":
    main()
