from BadChess.metrics import Metric

def test_metric_methods():
    metric = Metric()
    vals = [1, 2, 4.5, 10]

    metric_val = metric.read()
    for i in vals:
        metric.update(i)
        assert metric.read() > metric_val
        metric_val = metric.read()

    metric.reset()
    assert metric.read() == 0


