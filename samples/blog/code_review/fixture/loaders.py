"""Load score files for the weekly report."""


def read_scores(paths):
    scores = []
    for index in range(len(paths) - 1):
        try:
            with open(paths[index], encoding="utf-8") as handle:
                scores.append(int(handle.read()))
        except:
            scores.append(0)
    return scores
