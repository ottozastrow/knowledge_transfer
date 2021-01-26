import wandb
import argparse
import latency_tools
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--run_ids", type=str, default=None,
                    help="string of multiple ids separated by ' '")
parser.add_argument("--sweepid", type=str, default=None)
parser.add_argument("--query", type=str, default=None)
parser.add_argument("--filter_runs", action="store_true",
                    help="only run if miou>0.93 and no latency available")
args = parser.parse_args()


def update_bestvalmiou(run):
    hist = run.history()
    chosen_metric = "val_miou_score"
    if chosen_metric in hist.keys():
        metric_hist = hist[chosen_metric]
    else:
        print("metric not found", hist.keys())
        return
    try:
        miou = max(metric_hist)
    except Exception as e:
        print("could not get max from metric history")
        print(e)
        print(run.summary[chosen_metric])
        return

    if miou:
        run.summary['best_val_miou'] = miou
        run.summary.update()
        print("updated to ", miou)
    return miou


def filter_runs(runs):
    target_list = []
    for run in runs:
        summary_keys = run.summary.keys()
        if '_runtime' in summary_keys and "best_val_miou" in summary_keys:
                # and 'latency' not in summary_keys:
            if int(run.summary['_runtime']) > 900:
                try:
                    if float(run.summary['best_val_miou']) >= 0.91:
                        print(run.name)
                        print()
                        target_list.append(run)
                except Exception as e:
                    print(e)
                    print("while comparing ", run.summary['best_val_miou'])
                # best_miou = update_bestvalmiou(run)
    return target_list


if __name__ == "__main__":
    api = wandb.Api()
    runs = []
    if args.sweepid is not None:
        runs.extend(api.sweep(args.sweepid).runs)
    elif args.run_ids is not None:
        queries = args.run_ids.split(" ")
        for query in queries:
            runs.append(api.run(query))
    elif args.query is not None:
        runs.extend(api.runs(args.query))
    else:
        raise Exception("invalid args")
    print("found: ", len(runs), "wandb runs")

    if args.filter_runs:
        runs = filter_runs(runs)
        print("len of runs after filtering is ", len(runs))
    for run in tqdm.tqdm(runs):
        print(run)
        print("run ", run.name)
        if "best_val_miou" in run.summary.keys():
            print(run.summary['best_val_miou'])
        try:
            latency_tools.call_raspberrypi(run.config, run)
        except Exception as e:
            print(e)
