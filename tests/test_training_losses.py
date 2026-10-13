import torch

from scripts.training.Active.train_all_specialists import (
    NegBinomRankingLoss,
    REGION_CONFIGS,
    calculate_batch_metrics,
    negative_binomial_from_mean,
)


def test_negative_binomial_is_parameterized_by_requested_mean():
    mu = torch.tensor([0.25, 2.0, 8.0], dtype=torch.float32)
    r = torch.tensor(5.0, dtype=torch.float32)

    distribution = negative_binomial_from_mean(mu, r)

    torch.testing.assert_close(distribution.mean, mu)


def test_negative_binomial_ranking_loss_is_finite_and_differentiable():
    criterion = NegBinomRankingLoss(
        nb_r_init=5.0,
        ranking_weight=3.0,
        indecision_weight=0.2,
    )
    prediction = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 2.0, 5.0])

    loss = criterion(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert criterion.log_r.grad is not None
    assert torch.isfinite(criterion.log_r.grad)
    assert criterion.last_components['total'] == loss.detach().item()
    assert criterion.last_components['nb_nll'] > 0


def test_negative_binomial_dispersion_is_constrained_after_optimizer_step():
    criterion = NegBinomRankingLoss(nb_r_init=2.0, nb_r_min=0.75, nb_r_max=10.0)
    with torch.no_grad():
        criterion.log_r.fill_(-100.0)

    criterion.constrain_dispersion_()

    torch.testing.assert_close(criterion.dispersion(), torch.tensor(0.75))


def test_negative_binomial_rejects_invalid_dispersion_bounds():
    try:
        NegBinomRankingLoss(nb_r_init=0.5, nb_r_min=0.75)
    except ValueError as exc:
        assert 'nb_r_min' in str(exc)
    else:
        raise AssertionError('Limites inválidos deveriam falhar')


def test_fortaleza_regularization_preserves_incumbent_checkpoint_floor():
    config = REGION_CONFIGS['fortaleza']

    assert config['incumbent_validation_score'] == 0.42734
    assert config['validation_min_delta'] == 0.0025
    assert config['nb_r_min'] == 0.75
    assert config['nb_r_lr'] <= 0.00005
    assert config['patience'] == 6


def test_batch_metrics_report_hits_misses_false_positives_and_count_errors():
    prediction = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    target = torch.tensor([2.0, 0.0, 1.0, 0.0, 0.0])

    metrics = calculate_batch_metrics(prediction, target, 'negbinom')

    assert metrics['target_positive_nodes'] == 2
    assert metrics['target_total_events'] == 3.0
    assert metrics['hits_at_10'] == 2
    assert metrics['misses_at_10'] == 0
    assert metrics['false_positives_at_10'] == 3
    assert metrics['precision_at_10'] == 1.0
    assert metrics['error_scale'] == 'count_mean'
    assert metrics['mae'] >= 0
    assert metrics['rmse'] >= metrics['mae']
