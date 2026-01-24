"""helper script to plot comparison between different models/personas"""
import json

import matplotlib.pyplot as plt

def main() -> None:
    """main function"""
    model_str = "mdq100-Gemma3-Instruct-Abliterated"

    # load the results
    results_1 = json.load(
        open(
            f"logs/{model_str}-12b_safety_questions_pass@1.json", "r", encoding="utf-8"
        )
    )
    results_2 = json.load(
        open(
            f"logs/{model_str}-12b_safety_questions_pass@5.json", "r", encoding="utf-8"
        )
    )
    results_3 = json.load(
        open(
            f"logs/{model_str}-27b_safety_questions_pass@1.json", "r", encoding="utf-8"
        )
    )
    results_4 = json.load(
        open(
            f"logs/{model_str}-27b_safety_questions_pass@5.json", "r", encoding="utf-8"
        )
    )

    # plot the results with each model per persona
    personalities = list(results_1.keys())
    scores_1 = list(results_1.values())
    scores_2 = list(results_2.values())
    scores_3 = list(results_3.values())
    scores_4 = list(results_4.values())

    x = range(len(personalities))
    width = 0.2
    _, ax = plt.subplots(figsize=(36, 18))
    ax.bar(
        [i - 1.5 * width for i in x],
        scores_1,
        width,
        label="12B pass@1",
        color="b",
    )
    ax.bar(
        [i - 0.5 * width for i in x],
        scores_2,
        width,
        label="12B pass@5",
        color="c",
    )
    ax.bar(
        [i + 0.5 * width for i in x],
        scores_3,
        width,
        label="27B pass@1",
        color="orange",
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        scores_4,
        width,
        label="27B pass@5",
        color="r",
    )
    ax.set_xticks(x, personalities, rotation=45, ha="right")
    ax.set_xlabel("Personalities")
    ax.set_ylabel("Correct Answers (%)")
    ax.set_title(f"Safety Questions Test Results Comparison - {model_str}")
    ax.set_ylim(0, 100)
    ax.legend()
    #plt.tight_layout()
    plt.savefig(f"logs/{model_str}_safety_questions_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
