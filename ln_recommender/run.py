from ln_recommender.files import get_sources, save_csv
from ln_recommender.cli import get_inputs
from ln_recommender.parse import get_freq, parse_text
from ln_recommender.estimate import read_data, load_model, read_data_cluster
import os


def execute_on_inputs():
    inputs = get_inputs()
    if inputs.command == "eval":
        sources = get_sources(inputs)
        freq = get_freq(inputs.freq)
        if inputs.model_load is not None:
            model = load_model("model.cbm")
        else:
            model = read_data(inputs.training_data, inputs.eval_mode, inputs.model_save)

        data = []
        for path, filename in sources:
            out = parse_text(os.path.join(path, filename), filename, freq)
            pred_list = [
                out.p70,
                out.p80,
                out.p90,
                out.p95,
                out.p99,
                out.avg,
                out.median,
                out.mode,
                out.verb,
                out.aux,
            ]
            pred = model.predict(pred_list)
            csv_data = pred_list
            csv_data.append(pred[0])
            csv_data.append(filename)

            if inputs.output_csv is not None:
                data.append(csv_data)
            print_stats(filename, out, pred[0])

        if inputs.output_csv is not None:
            save_csv("out.csv", data)

    elif inputs.command == "train":
        model = read_data(inputs.training_data, inputs.eval_mode, inputs.model_save)

    elif inputs.command == "cluster":
        model = read_data_cluster(
            inputs.training_data, inputs.min_cluster_size, inputs.min_samples
        )


def print_stats(filename, stat, prediction):
    print("")
    print(f"Filename: {filename}")
    print("Use the line below with difficulty label to add new training data")
    print(
        f"{stat.p70:.2f}"
        f",{stat.p80:.2f}"
        f",{stat.p90:.2f}"
        f",{stat.p95:.2f}"
        f",{stat.p99:.2f}"
        f",{stat.avg:.2f}"
        f",{stat.median:.2f}"
        f",{stat.mode:.2f}"
        f",{stat.verb:.2f}"
        f",{stat.aux:.2f}"
    )
    print(f"Readability: {prediction}")
    print(f"Readability 70% frequency: {stat.p70:.0f}")
    print(f"Readability 80% frequency: {stat.p80:.0f}")
    print(f"Readability 90% frequency: {stat.p90:.0f}")
    print(f"Readability 95% frequency: {stat.p95:.0f}")
    print(f"Readability 99% frequency: {stat.p99:.0f}")
    print(f"Avg Sentence Length: {stat.avg:.2f}")
    print(f"Median Sentence Length: {stat.median:.2f}")
    print(f"Mode Sentence Length: {stat.mode:.2f}")
    print(f"Avg Verb Count: {stat.verb:.2f}")
    print(f"Avg Auxiliary Verb Count: {stat.aux:.2f}")
